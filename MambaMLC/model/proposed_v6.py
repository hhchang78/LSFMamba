import torch
import torch.nn as nn
import copy
import math
import numpy as np
from einops import repeat
from fvcore.nn import flop_count
from torchsummary import summary

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref


class SCANS(nn.Module):
    def __init__(self, size=16, dim=2, scan_type='scan', ):
        super().__init__()
        size = int(size)
        max_num = size ** dim
        indexes = np.arange(max_num)
        if 'sweep' == scan_type:  # ['sweep', 'zigzag', 'spiral']
            locs_flat = indexes
        elif 'zigzag' == scan_type:
            indexes = indexes.reshape(size, size)
            locs_flat = []
            for i in range(2 * size - 1):
                if i % 2 == 0:
                    start_col = max(0, i - size + 1)
                    end_col = min(i, size - 1)
                    for j in range(start_col, end_col + 1):
                        locs_flat.append(indexes[i - j, j])
                else:
                    start_row = max(0, i - size + 1)
                    end_row = min(i, size - 1)
                    for j in range(start_row, end_row + 1):
                        locs_flat.append(indexes[j, i - j])
            locs_flat = np.array(locs_flat)
        elif 'spiral' == scan_type:
            locs_flat = self._generate_spiral_indices(size, indexes)
        elif 'random' == scan_type:
            locs_flat = np.random.permutation(indexes)
        else:
            raise Exception('invalid encoder mode')
        locs_flat_inv = np.argsort(locs_flat)
        index_flat = torch.LongTensor(locs_flat.astype(np.int64)).unsqueeze(0).unsqueeze(1)
        index_flat_inv = torch.LongTensor(locs_flat_inv.astype(np.int64)).unsqueeze(0).unsqueeze(1)
        self.index_flat = nn.Parameter(index_flat, requires_grad=False)
        self.index_flat_inv = nn.Parameter(index_flat_inv, requires_grad=False)

    def _generate_spiral_indices(self, size, indexes):
        locs_flat = []
        indexes = indexes.reshape(size, size)
        # 初始化边界
        top, bottom = 0, size - 1
        left, right = 0, size - 1
        while top <= bottom and left <= right:
            # 从左到右遍历顶部边界
            for col in range(left, right + 1):
                locs_flat.append(indexes[top, col])
            top += 1
            # 从上到下遍历右侧边界
            for row in range(top, bottom + 1):
                locs_flat.append(indexes[row, right])
            right -= 1
            # 从右到左遍历底部边界（如果还没遍历完）
            if top <= bottom:
                for col in range(right, left - 1, -1):
                    locs_flat.append(indexes[bottom, col])
                bottom -= 1
            # 从下到上遍历左侧边界（如果还没遍历完）
            if left <= right:
                for row in range(bottom, top - 1, -1):
                    locs_flat.append(indexes[row, left])
                left += 1
        locs_flat = np.array(locs_flat)
        return locs_flat

    def __call__(self, img):
        img_encode = self.encode(img)
        return img_encode

    def encode(self, img):
        img_encode = torch.zeros(img.shape, dtype=img.dtype, device=img.device).scatter_(2, self.index_flat_inv.expand(
            img.shape), img)
        return img_encode

    def decode(self, img):
        img_decode = torch.zeros(img.shape, dtype=img.dtype, device=img.device).scatter_(2, self.index_flat.expand(
            img.shape), img)
        return img_decode


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            ssm_ratio=1,  # 2
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            device=None,
            dtype=None,
            size=9,
            scan_type='scan',
            num_direction=4,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.ssm_ratio = ssm_ratio
        self.d_inner = int(self.ssm_ratio * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.num_direction = num_direction

        x_proj_weight = [nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs).weight
                         for _ in range(self.num_direction)]
        self.x_proj_weight = nn.Parameter(torch.stack(x_proj_weight, dim=0))
        dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.num_direction)]
        self.dt_projs_weight = nn.Parameter(torch.stack([dt_proj.weight for dt_proj in dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([dt_proj.bias for dt_proj in dt_projs], dim=0))

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.num_direction, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=self.num_direction, merge=True)  # (K=4, D, N)

        self.scans = SCANS(size=size, scan_type=scan_type)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        B, C, H, W = x.shape
        L = H * W
        K = self.num_direction
        if K == 1:
            xs = self.scans.encode(x.view(B, -1, L))
        else:
            xs = []
            if K >= 2:
                xs.append(self.scans.encode(x.view(B, -1, L)))
            if K >= 4:
                xs.append(self.scans.encode(torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)))
            if K >= 8:
                xs.append(self.scans.encode(torch.rot90(x, k=1, dims=(2, 3)).contiguous().view(B, -1, L)))
                xs.append(self.scans.encode(
                    torch.transpose(torch.rot90(x, k=1, dims=(2, 3)), dim0=2, dim1=3).contiguous().view(B, -1, L)))
            xs = torch.stack(xs, dim=1).view(B, K // 2, -1, L)
            xs = torch.cat([xs, torch.flip(xs, dims=[-1])], dim=1)
            # xs = torch.stack(xs, dim=1).view(B, K, -1, L)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        if K == 1:
            y = self.scans.decode(out_y[:, 0])
        else:
            inv_y = torch.flip(out_y[:, K // 2:K], dims=[-1]).view(B, K // 2, -1, L)
            ys = []
            if K >= 2:
                ys.append(self.scans.decode(out_y[:, 0]))
                ys.append(self.scans.decode(inv_y[:, 0]))
            if K >= 4:
                ys.append(
                    torch.transpose(self.scans.decode(out_y[:, 1]).view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(
                        B, -1, L))
                ys.append(
                    torch.transpose(self.scans.decode(inv_y[:, 1]).view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(
                        B, -1, L))
            if K >= 8:
                ys.append(
                    torch.rot90(self.scans.decode(out_y[:, 2]).view(B, -1, W, H), k=3,
                                dims=(2, 3)).contiguous().view(B, -1, L))
                ys.append(
                    torch.rot90(self.scans.decode(inv_y[:, 2]).view(B, -1, W, H), k=3,
                                dims=(2, 3)).contiguous().view(B, -1, L))
                ys.append(
                    torch.rot90(torch.transpose(self.scans.decode(out_y[:, 3]).view(B, -1, W, H), dim0=2, dim1=3), k=3,
                                dims=(2, 3)).contiguous().view(B, -1, L))
                ys.append(
                    torch.rot90(torch.transpose(self.scans.decode(inv_y[:, 3]).view(B, -1, W, H), dim0=2, dim1=3), k=3,
                                dims=(2, 3)).contiguous().view(B, -1, L))
            y = sum(ys)
        return y
    def forward(self, x: torch.Tensor, **kwargs):
        y = self.forward_core(x)
        return y


class SSM(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            ssm_ratio=1,  # 2
            # d_conv=3,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # dropout=0.,
            # conv_bias=True,
            # bias=False,
            device=None,
            dtype=None,
            size=9,
            scan_type='scan',
            num_direction=2,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_conv = d_conv
        self.ssm_ratio = ssm_ratio
        self.d_inner = int(self.ssm_ratio * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.num_direction = num_direction

        x_proj_weight = [nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs).weight
                         for _ in range(self.num_direction)]
        self.x_proj_weight = nn.Parameter(torch.stack(x_proj_weight, dim=0))
        dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.num_direction)]
        self.dt_projs_weight = nn.Parameter(torch.stack([dt_proj.weight for dt_proj in dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([dt_proj.bias for dt_proj in dt_projs], dim=0))

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.num_direction, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=self.num_direction, merge=True)  # (K=4, D, N)

        # self.out_norm = nn.LayerNorm(self.d_inner)
        # self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        # self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self.scans = SCANS(size=size, scan_type=scan_type)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x1: torch.Tensor, x2: torch.Tensor):
        self.selective_scan = selective_scan_fn
        B, C, H, W = x1.shape
        L = H*W
        K = self.num_direction

        xs1 = self.scans.encode(x1.view(B, -1, L))
        xs2 = self.scans.encode(x2.view(B, -1, L))

        xs1 = xs1.view(B, C, 1, L)
        xs2 = xs2.view(B, C, 1, L)
        xs = torch.cat([xs1, xs2], dim=2)
        xs = xs.permute(0, 1, 3, 2).contiguous().view(B, C, -1)

        if K==1:
            xs = xs.view(B, K, -1, 2*L)
        elif K==2:
            xs = xs.view(B, K // 2, -1, 2*L)
            xs = torch.cat([xs, torch.flip(xs, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, 2 * L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, 2 * L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, 2 * L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, 2 * L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, 2 * L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, 2 * L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, 2 * L)
        assert out_y.dtype == torch.float

        out_y = out_y.view(B, K, -1, L, 2).permute(0, 1, 2, 4, 3).contiguous()

        out_y1, out_y2 = out_y.chunk(2, dim=3)
        out_y1 = out_y1[:,:,:,0]
        out_y2 = out_y2[:,:,:,0]

        if K == 1:
            y1 = self.scans.decode(out_y1[:, 0])
            y2 = self.scans.decode(out_y2[:, 0])
        else:
            inv_y1 = torch.flip(out_y1[:, K // 2:K], dims=[-1]).view(B, K // 2, -1, L)
            inv_y2 = torch.flip(out_y2[:, K // 2:K], dims=[-1]).view(B, K // 2, -1, L)
            ys1 = [self.scans.decode(out_y1[:, 0]), self.scans.decode(inv_y1[:, 0])]
            ys2 = [self.scans.decode(out_y2[:, 0]), self.scans.decode(inv_y2[:, 0])]
            y1 = sum(ys1)
            y2 = sum(ys2)
        return y1, y2

    def forward(self, x1: torch.Tensor, x2: torch.Tensor,**kwargs):
        y = self.forward_core(x1, x2)
        return y


# =====================================================
class SpatialMamba(nn.Module):
    """
    mamba module in spatial
    """

    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2,
            dt_rank="auto",
            # dw-conv ===============
            d_conv=1,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            size=9,
            scan_type="scan",
            num_direction=4,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # sigma
        self.d_conv = d_conv

        # in norm =======================================
        self.in_norm = nn.LayerNorm(self.d_model)

        # in proj =======================================
        d_proj = int(self.d_inner * 2)
        self.in_proj = nn.Linear(self.d_model, d_proj, bias=bias, **factory_kwargs)

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
        else:
            self.conv2d = nn.Identity()
        self.act = nn.SiLU()

        self.conv3_1 = nn.Sequential(
            nn.Conv2d(self.d_inner, self.d_inner, (3, 1), stride=1, padding=(1, 0), groups=self.d_inner),
            nn.BatchNorm2d(self.d_inner),
            nn.GELU(),
            nn.Conv2d(self.d_inner, self.d_inner, (1, 3), stride=1, padding=(0, 1), groups=self.d_inner),
            nn.BatchNorm2d(self.d_inner),
            nn.GELU(), )

        self.conv5_1 = nn.Sequential(
            nn.Conv2d(self.d_inner, self.d_inner, (5, 1), stride=1, padding=(2, 0), groups=self.d_inner),
            nn.BatchNorm2d(self.d_inner),
            nn.GELU(),
            nn.Conv2d(self.d_inner, self.d_inner, (1, 5), stride=1, padding=(0, 2), groups=self.d_inner),
            nn.BatchNorm2d(self.d_inner),
            nn.GELU(), )

        self.conv2 = nn.Sequential(
            nn.Conv2d(2 * self.d_inner, self.d_inner, 1),
            nn.BatchNorm2d(self.d_inner),
            nn.GELU(), )

        # SSM =======================================
        self.ssm = SS2D(
            d_model=self.d_model,
            d_state=self.d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            size=size,
            scan_type=scan_type,
            num_direction=num_direction,
            **kwargs,
        )

        # out norm =======================================
        self.out_norm = nn.LayerNorm(self.d_inner)

        # out proj =======================================
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward(self, x: torch.Tensor):
        B, H, W, D = x.shape
        x_res = x  # (b, h, w, d)
        x = self.in_norm(x)
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
        z= self.act(z)
        z = z.permute(0, 3, 1, 2).contiguous()  # (b, d, h, w)
        z = torch.cat((self.conv3_1(z), self.conv5_1(z)), dim=1)
        z = self.conv2(z)
        x = x.permute(0, 3, 1, 2).contiguous()  # (b, d, h, w)
        # x = torch.cat((self.conv3_1(x), self.conv5_1(x)), dim=1)
        # x = self.conv2(x)

        if self.d_conv > 1:
            x = self.act(self.conv2d(x))  # (b, d, h, w)
        else:
            ...

        y = self.ssm(x)  # (b, d, h, w) -> (b, d, -1)
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        z = z.permute(0, 2, 3, 1).contiguous()  # (b, h, w, d)
        # y = torch.cat((y, z), dim=-1)
        y = y * z
        y = self.out_proj(y) + x_res
        out = self.dropout(y)
        return out


# =====================================================
class CrossFusionMamba(nn.Module):
    """
    Multimodal Cross Fusion Mamba with SSM or VSSM
    """

    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2,
            dt_rank="auto",
            # dw-conv ===============
            d_conv=1,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            size=9,
            scan_type="scan",
            num_direction=4,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # sigma
        self.d_conv = d_conv

        # in proj =======================================
        d_proj = self.d_inner
        self.in_proj_hsi = nn.Linear(self.d_model, d_proj, bias=bias, **factory_kwargs)
        self.in_proj_lidar = nn.Linear(self.d_model, d_proj, bias=bias, **factory_kwargs)

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d_hsi = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.conv2d_lidar = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
        self.act = nn.SiLU()

        # SSM =======================================
        self.ssm = SSM(
            d_model=self.d_model,
            d_state=self.d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            size=size,
            scan_type=scan_type,
            num_direction=num_direction,
            **kwargs,
        )

        # out norm =======================================
        self.out_norm_hsi = nn.LayerNorm(self.d_inner)
        self.out_norm_lidar = nn.LayerNorm(self.d_inner)

        # out proj =======================================
        self.out_proj_hsi = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj_lidar = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward(self, x_hsi: torch.Tensor, x_lidar: torch.Tensor):  # (b, h, w, d)

        B, H, W, D = x_hsi.shape

        x_hsi_res = x_hsi  # for residual (b, h, w, d)
        x_lidar_res = x_lidar

        x_hsi = self.in_proj_hsi(x_hsi)
        x_lidar = self.in_proj_lidar(x_lidar)

        x_hsi = x_hsi.permute(0, 3, 1, 2).contiguous()
        x_lidar = x_lidar.permute(0, 3, 1, 2).contiguous()

        if self.d_conv > 1:
            x_hsi = self.act(self.conv2d_hsi(x_hsi))  # (b, d, h, w)
            x_lidar = self.act(self.conv2d_lidar(x_lidar))
        else:
            ...

        y_hsi, y_lidar = self.ssm(x_hsi, x_lidar) # (b, d, h, w) -> (b, d, -1)

        y_hsi = torch.transpose(y_hsi, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y_lidar = torch.transpose(y_lidar, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y_hsi = self.out_norm_hsi(y_hsi)
        y_lidar = self.out_norm_lidar(y_lidar)

        y_hsi = self.out_proj_hsi(y_hsi) + x_hsi_res
        y_lidar = self.out_proj_lidar(y_lidar) + x_lidar_res  # (b, h, w, d)
        y_hsi = self.dropout(y_hsi)
        y_lidar = self.dropout(y_lidar)
        return y_hsi, y_lidar


# fvcore flops =======================================
def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    assert not with_complex
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


def print_jit_input_names(inputs):
    # tensor.11, dt.1, A.1, B.1, C.1, D.1, z.1, None
    try:
        print("input params: ", end=" ", flush=True)
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)


def selective_scan_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)

    # # xs, dts, As, Bs, Cs, Ds (skip), z (skip), dt_projs_bias (skip)
    # assert inputs[0].debugName().startswith("xs")  # (B, D, L)
    # assert inputs[1].debugName().startswith("dts")  # (B, D, L)
    # assert inputs[2].debugName().startswith("As")  # (D, N)
    # assert inputs[3].debugName().startswith("Bs")  # (D, N)
    # assert inputs[4].debugName().startswith("Cs")  # (D, N)
    with_Group = len(inputs[3].type().sizes()) == 4
    with_D = inputs[5].debugName().startswith("Ds")
    if not with_D:
        with_z = len(inputs) > 5 and inputs[5].debugName().startswith("z")
    else:
        with_z = len(inputs) > 6 and inputs[6].debugName().startswith("z")
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=with_D, with_Z=with_z, with_Group=with_Group)
    # flops = flops_selective_scan_ref(B=B, L=L, D=D, N=N, with_D=with_D, with_Z=with_z, with_Group=with_Group)
    return flops


# =====================================================
class Proposed(nn.Module):
    def __init__(self, dataset_name, hidden_dim=64):
        super(Proposed, self).__init__()

        if dataset_name == 'Houston2013':
            hsi_dim = 144
            lidar_dim = 1
            num_classes = 15
            patch_size = 11
        elif dataset_name == 'MUUFL':
            hsi_dim = 64
            lidar_dim = 2
            num_classes = 11
            patch_size = 11
        elif dataset_name == 'GRSS07':
            hsi_dim = 6
            lidar_dim = 1
            num_classes = 5
            patch_size = 11
        elif dataset_name == 'Trento':
            hsi_dim = 63
            lidar_dim = 1
            num_classes = 6
            patch_size = 11
        elif dataset_name == 'Berlin':
            hsi_dim = 244
            lidar_dim = 4
            num_classes = 8
            patch_size = 11
        elif dataset_name == 'Augsburg':
            hsi_dim = 180
            lidar_dim = 4  # sar 4 lidar 1
            num_classes = 7
            patch_size = 11
        elif dataset_name == 'Houston2018':
            hsi_dim = 50
            lidar_dim = 1
            num_classes = 20
            patch_size = 11
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}. Dataset does not exist.")

        self.hsi_proj = nn.Sequential(
            nn.Conv2d(in_channels=hsi_dim, out_channels=hidden_dim, kernel_size=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        self.lidar_proj = nn.Sequential(
            nn.Conv2d(in_channels=lidar_dim, out_channels=hidden_dim, kernel_size=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        scan_type = 'spiral'
        print(scan_type)
        num_dir = 2
        self.spatial_hsi1 = SpatialMamba(scan_type=scan_type, d_model=hidden_dim, size=patch_size,
                                           ssm_ratio=1, num_direction=num_dir)
        self.spatial_lidar1 = SpatialMamba(scan_type=scan_type, d_model=hidden_dim, size=patch_size,
                                           ssm_ratio=1, num_direction=num_dir)

        self.spatial_hsi2 = SpatialMamba(scan_type=scan_type, d_model=hidden_dim, size=patch_size-4,
                                           ssm_ratio=1, num_direction=num_dir)
        self.spatial_lidar2 = SpatialMamba(scan_type=scan_type, d_model=hidden_dim, size=patch_size-4,
                                           ssm_ratio=1, num_direction=num_dir)

        self.fusion_mamba1 = CrossFusionMamba(d_model=hidden_dim, ssm_ratio=1, size=patch_size, scan_type=scan_type,
                                             num_direction=2)
        self.fusion_mamba2 = CrossFusionMamba(d_model=hidden_dim, ssm_ratio=1, size=patch_size-4, scan_type=scan_type,
                                             num_direction=2)

        self.avp = nn.AdaptiveAvgPool2d(1)
        self.class_head = nn.Linear(in_features=int(hidden_dim * 2), out_features=num_classes)

    def forward(self, x_hsi, x_lidar):

        # in_proj
        x_hsi = self.hsi_proj(x_hsi)
        x_lidar = self.lidar_proj(x_lidar)
        x1 = x_hsi.permute(0, 2, 3, 1).contiguous()  # (b, h, w, d)
        x2 = x_lidar.permute(0, 2, 3, 1).contiguous()

        x1 = self.spatial_hsi1(x1)
        x2 = self.spatial_lidar1(x2)
        x1, x2 = self.fusion_mamba1(x1, x2)

        x1 = x1[:, 2:-2, 2:-2, :]
        x2 = x2[:, 2:-2, 2:-2, :]

        x1 = self.spatial_hsi2(x1)
        x2 = self.spatial_lidar2(x2)
        x1, x2 = self.fusion_mamba2(x1, x2)

        x_out = torch.cat((x1, x2), dim=-1)
        x_out = x_out.permute(0, 3, 1, 2).contiguous()

        # classifier
        x_out = self.avp(x_out)
        out = self.class_head(x_out.squeeze(2).squeeze(2))
        return out

    def flops(self, shape1=(144, 11, 11), shape2=(1, 11, 11)):
        supported_ops = {
            "aten::silu": None,  # as relu is in _IGNORED_OPS
            "aten::neg": None,  # as relu is in _IGNORED_OPS
            "aten::exp": None,  # as relu is in _IGNORED_OPS
            "aten::flip": None,  # as permute is in _IGNORED_OPS
            "aten::leaky_relu": None,  # as relu is in _IGNORED_OPS

            "aten::mul": None,  # ???????????????
            "aten::add": None,  # ???????????????
            "aten::gelu": None,
            "aten::scatter_": None,

            "prim::PythonOp.CrossScan": None,
            "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit,
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input1 = torch.randn((1, *shape1), device=next(model.parameters()).device)
        input2 = torch.randn((1, *shape2), device=next(model.parameters()).device)
        # params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input1, input2), supported_ops=supported_ops)

        del model, input1, input2
        return sum(Gflops.values()) * 1e3
    # return f"params {params * 4 /(1024 ** 2)} GFLOPs {sum(Gflops.values()) * 1e3}"


if __name__ == '__main__':
    import time
    from torchinfo import summary

    model = Proposed('Houston2013').cuda()
    input_size = [(1, 144, 11, 11), (1, 1, 11, 11)]
    start_time = time.time()
    summary(model, input_size)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"运行时间: {elapsed_time:.6f} 秒")
    print(model.flops())


