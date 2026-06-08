import numpy as np
import matplotlib.pyplot as plt


def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi,
                        ground_truth.shape[0] * 2.0 / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)
    return 0


def list_to_colormap(x_list, dataset_name):
    y = np.zeros((x_list.shape[0], 3))
    if dataset_name == 'Houston2013':
        # colors = np.array([
        #     [0, 0, 0],
        #     [84, 172, 71],
        #     [108, 147, 67],
        #     [67, 193, 54],
        #     [60, 124, 70],
        #     [146, 78, 52],
        #     [105, 160, 198],
        #     [198, 177, 193],
        #     [128, 128, 128],
        #     [127, 0, 0],
        #     [127, 127, 0],
        #     [255, 166, 0],
        #     [127, 0, 127],
        #     [0, 127, 127],
        #     [0, 0, 127],
        #     [188, 0, 31]
        # ]) / 255.
        colors = np.array([
            [0, 0, 0],  # Background
            [76, 175, 80],  # Healthy grass
            [139, 195, 74],  # Stressed grass
            [0, 200, 83],  # Synthetic grass
            [34, 100, 34],  # Trees
            [166, 97, 26],  # Soil
            [65, 182, 230],  # Water
            [204, 121, 167],  # Residential
            [160, 160, 160],  # Commercial
            [190, 30, 45],  # Road
            [230, 159, 0],  # Highway
            [240, 228, 66],  # Railway
            [142, 68, 173],  # Parking Lot 1
            [0, 150, 136],  # Parking Lot 2
            [0, 82, 204],  # Tennis Court
            [208, 28, 139]  # Running Track
        ], dtype=float) / 255.0
    if dataset_name == 'Berlin':
        colors = np.array([
            [0, 0, 0],
            [26,163,25],
            [16,216,116],
            [216,89,89],
            [210, 181, 0],
            [204, 102, 204],
            [173, 216, 230],
            [180, 142, 173],
            [191, 97, 106],
            [163, 190, 140],
            [235, 203, 139],
            [127, 127, 0],
        ]) / 255.
    if dataset_name == 'MUUFL':
        colors = np.array([
            [0, 0, 0],
            [26,163,25],
            [216,89,89],
            [210, 181, 0],
            [204, 102, 204],
            [173, 216, 230],
            [173, 216, 230],
            [88, 150, 32],
            [54, 43, 17],
            [154, 67, 89],
            [34, 189, 70],
            [123, 24, 57],
        ]) / 255.
    if dataset_name == 'Trento':
        colors = np.array([
            [0, 0, 0],
            [102, 187, 106],
            [230, 74, 25],
            [210, 180, 140],
            [46, 125, 50],
            [156, 39, 176],
            [120, 120, 120]
        ]) / 255.0
    if dataset_name == 'Augsburg':
        # colors = np.array([
        #     [0, 0, 0],
        #     [26,163,25],
        #     [216,216,216],
        #     [216,89,89],
        #     [204, 102, 204],
        #     [204,153,52],
        #     [244,231,1],
        #     [0, 53, 255],
        # ]) / 255.
        colors = np.array([
            [0, 0, 0],  # Background
            [46, 125, 50],  # Forest
            [244, 67, 54],  # Residential Area
            [120, 120, 120],  # Industrial Area
            [156, 204, 101],  # Low Plants
            [255, 202, 40],  # Allotment
            [171, 71, 188],  # Commercial Area
            [33, 150, 243]  # Water
        ], dtype=float) / 255.0
    for index, item in enumerate(x_list):
        y[index] = colors[int(item),:]
    return y


def generate_png(total_iter, net, gt_hsi, device, total_indices, dataset_name, path):
    pred_test = []
    for X1, X2, y in total_iter:
        X1 = X1.to(device)
        X2 = X2.to(device)
        net.eval()
        batch_out = net(X1, X2)
        # batch_out = net(X1, X2)[0]
        pred_test.extend(batch_out.cpu().argmax(axis=1).detach().numpy()+1)
    gt = gt_hsi.flatten()
    x_label = np.zeros(gt_hsi.shape)
    x_label[total_indices[:,0], total_indices[:,1]] = pred_test

    x = np.ravel(x_label)
    y_list = list_to_colormap(x, dataset_name)
    y_gt = list_to_colormap(gt, dataset_name)
    y_re = np.reshape(y_list, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    gt_re = np.reshape(y_gt, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    classification_map(y_re, gt_hsi, 900,
                       path + '.png')
    classification_map(gt_re, gt_hsi, 900,
                       path + '_gt.png')
    print('------Get classification maps successful-------')

# def generate_all_png(all_iter, net, gt_hsi, device, all_indices, dataset_name, path):
#     pred_test = []
#     for X1, X2, y in all_iter:
#         X1 = X1.to(device)
#         X2 = X2.to(device)
#         net.eval()
#         pred_test.extend(net(X1, X2).cpu().argmax(axis=1).detach().numpy())
#         # pred_test.extend(net(X1, X2)[0].cpu().argmax(axis=1).detach().numpy())
#     gt = gt_hsi.flatten()
#     x_label = np.zeros(gt_hsi.shape)
#     x_label[all_indices[:, 0], all_indices[:, 1]] = pred_test
#     for i in range(len(gt)):
#         if gt[i] == 0:
#             gt[i] = 16
#             # x_label[i] = 16
#     gt = gt[:] - 1
#
#     x = np.ravel(x_label)
#     y_list = list_to_colormap(x, dataset_name)
#     y_gt = list_to_colormap(gt, dataset_name)
#     y_re = np.reshape(y_list, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
#     gt_re = np.reshape(y_gt, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
#     classification_map(y_re, gt_hsi, 300,
#                        path + '.eps')
#     classification_map(y_re, gt_hsi, 300,
#                        path + '.png')
#     classification_map(gt_re, gt_hsi, 300,
#                        path + '_gt.png')
#     print('------Get classification maps successful-------')
