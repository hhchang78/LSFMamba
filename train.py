import os
import gc
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import time

def train(args, model, optimizer, criterion, train_loader, test_loader, device, scheduler=None):

    gc.collect()
    torch.cuda.empty_cache()

    # Reset GPU memory statistics
    torch.cuda.reset_peak_memory_stats(device)

    best_acc = 0.
    for epoch in range(args.epoch):

        train_start_time = time.time()

        model.train()

        losses = AverageMeter()
        tar = np.array([])
        pre = np.array([])
        for batch_idx, (hsi, lidar, batch_target) in enumerate(train_loader):

            hsi, lidar, batch_target = hsi.to(device), lidar.to(device), batch_target.to(device)

            batch_out = model(hsi, lidar)
            loss = criterion(batch_out, batch_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            losses.update(loss.data, batch_target.shape[0])
            batch_pred = np.argmax(batch_out.detach().cpu().numpy(), axis=1)
            batch_target = batch_target.detach().cpu().numpy()

            tar = np.append(tar, batch_pred)
            pre = np.append(pre, batch_target)

        train_end_time = time.time()
        training_time = train_end_time - train_start_time

        if (epoch % args.test_freq == 0) | (epoch == args.epoch - 1):
            test_start_time = time.time()
            test_loss, results = validation(model, criterion, test_loader, device)
            aa, oa, kappa = results['AA'], results['OA'], results['Kappa']
            test_end_time = time.time()
            test_time = test_end_time - test_start_time

            is_best = aa >= best_acc
            best_acc = max(aa, best_acc)
            save_checkpoint(model, is_best, args, epoch)

            print(f'Epoch: {epoch}, train_time: {training_time:.4f}, test_time: {test_time:.4f}, '
                  f'OA: {oa:.4f}, AA: {aa:.4f}, Kappa: {kappa:.4f}, best_acc: {best_acc:.4f}')
            print(results['PA'])

    peak_memory_used = torch.cuda.max_memory_allocated(device)
    print(f"Peak GPU memory used: {peak_memory_used / (1024 ** 2):.2f} MB")



def validation(model, criterion, test_loader, device):

    model.eval()
    with torch.no_grad():
        losses = AverageMeter()
        tar = np.array([])
        pre = np.array([])
        for batch_idx, (hsi, lidar, batch_target) in enumerate(test_loader):

            hsi, lidar, batch_target = hsi.to(device), lidar.to(device), batch_target.to(device)

            batch_out = model(hsi, lidar)
            loss = criterion(batch_out, batch_target)

            losses.update(loss.data, batch_target.shape[0])
            batch_pred = np.argmax(batch_out.detach().cpu().numpy(), axis=1)
            batch_target = batch_target.detach().cpu().numpy()

            tar = np.append(tar, batch_pred)
            pre = np.append(pre, batch_target)

        total_loss = losses.avg
        results = compute_metrics(tar, pre)

    return total_loss, results


def save_checkpoint(network, is_best, args, epoch):
    if not os.path.isdir(os.path.join(args.saving_path, args.dataset_name, args.model_name)):
        os.makedirs(os.path.join(args.saving_path, args.dataset_name, args.model_name), exist_ok=True)

    if is_best:
        # tqdm.write("epoch = {epoch}: best validation OA = {acc:.4f}".format(**kwargs))
        torch.save(network.state_dict(), os.path.join(args.saving_path, args.dataset_name, args.model_name, 'model_best.pth'))
    else:  # save the ckpt for each 5 epoch
        if epoch == args.epoch - 1:
            torch.save(network.state_dict(), os.path.join(args.saving_path, args.dataset_name, args.model_name, 'model.pth'))


class AverageMeter(object):

  def __init__(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def compute_metrics(pred, target):
    """Compute and print metrics (OA, PA, AA, Kappa)

    Args:
        pred: list of predicted labels
        target: list of target labels

    Returns:
        {Confusion Matrix, OA, PA, AA, Kappa}
    """

    results = {}

    # compute Overall Accuracy
    cm = confusion_matrix(target, pred)
    results['Confusion matrix'] = cm

    # compute Overall Accuracy (OA)
    oa = 1. * np.trace(cm) / np.sum(cm)
    results['OA'] = oa

    # compute Producer Accuracy (PA)
    n_classes = cm.shape[0]
    pa = np.array([1. * cm[i, i] / np.sum(cm[i, :]) for i in range(n_classes)])
    results['PA'] = pa

    # compute Average Accuracy (AA)
    aa = np.mean(pa)
    results['AA'] = aa

    # compute kappa coefficient
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(np.sum(cm) * np.sum(cm))
    kappa = (oa - pe) / (1 - pe)
    results['Kappa'] = kappa

    return results


