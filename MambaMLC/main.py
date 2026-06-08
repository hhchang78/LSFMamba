import argparse
import os
import random
import numpy as np
import torch
from pynvml import *

from utils.plot import generate_png
from utils.dataset import make_dataloader, make_dataloader_whole
from train import train, validation
from model.proposed_v6 import Proposed


parser = argparse.ArgumentParser(description="multi_modal RS classification")
parser.add_argument("--model_name", type=str, default='MambaMLC')
parser.add_argument("--dataset_name", type=str, default="Augsburg") # Augsburg Houston2013 MUUFL
parser.add_argument("--dataset_dir", type=str, default="./datasets")
parser.add_argument("--use_pca", type=bool, default=False)
parser.add_argument("--pca_component", type=int, default=30)
parser.add_argument("--patch_size", type=int, default=11)
parser.add_argument("--num_classes", type=int, default=7)

parser.add_argument("--epoch", type=int, default=150)
parser.add_argument("--warmup_epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=128) # 16
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--use_scheduler", type=bool, default=False)

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--saving_path", type=str, default="./model_params")
parser.add_argument("--is_train", type=bool, default=True)
parser.add_argument("--test_freq", type=int, default=10)


def setup_seed(seed):
    random.seed(seed)  # 设置Python内置的随机数生成器种子
    np.random.seed(seed)  # 设置NumPy随机数生成器种子
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)  # 设置PyTorch CPU随机数生成器种子
    torch.cuda.manual_seed(seed)  # 设置PyTorch GPU随机数生成器种子
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机数生成器种子（如果有多个GPU）
    torch.backends.cudnn.deterministic = True  # 确保每次调用返回相同的结果
    torch.backends.cudnn.benchmark = False  # 关闭优化

args = parser.parse_args()

setup_seed(args.seed)
print(args.seed)

# os.environ['CUDA_VISIBLE_DEVICES'] = args.device
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'

train_loader, test_loader = make_dataloader(args)

model = Proposed(args.dataset_name).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0)

criterion = torch.nn.CrossEntropyLoss()

if args.is_train:
    train(args, model, optimizer, criterion, train_loader, test_loader, device, scheduler)

else:
    # model.load_state_dict(torch.load(os.path.join(
    #     args.saving_path, args.dataset_name, args.model_name, 'model_best.pth')))
    test_loss, results = validation(model, criterion, test_loader, device)
    aa, oa, kappa = results['AA'], results['OA'], results['Kappa']
    print(f'OA: {oa:.4f}, AA: {aa:.4f}, Kappa: {kappa:.4f}')
    print(results['PA'])


# PLOT_CLS = True
# if PLOT_CLS:
#     model.load_state_dict(torch.load(os.path.join(
#         args.saving_path, args.dataset_name, args.model_name, 'model_best.pth')))
#
#     whole_loader, whole_indices, gt = make_dataloader_whole(args)
#
#     if not os.path.isdir(os.path.join('./cls_map', args.dataset_name)):
#         os.makedirs(os.path.join('./cls_map', args.dataset_name))
#     generate_png(
#         whole_loader, model, gt, device, whole_indices, args.dataset_name, os.path.join('./cls_map', args.dataset_name) + '/{}'.format(args.model_name))