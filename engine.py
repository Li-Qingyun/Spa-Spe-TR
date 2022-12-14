# TODO: ADD INFO

"""
Train and eval functions used in main.py
"""
import time

import numpy as np

import util
import datetime

from timm.utils import accuracy, AverageMeter
from timm.scheduler.scheduler import Scheduler

import torch
from torch.utils.data.dataloader import DataLoader

from util import false_logger
from sklearn import metrics
import scipy.io as sio
def train_one_epoch(
        model: torch.nn.Module,
        data_loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Scheduler,
        mixup_fn=None, max_norm: float = 0,
        epoch=None, total_epochs=None,
        logger=false_logger,):

    model.train()
    criterion.train()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    acc_meter = AverageMeter()
    acc_meter_spectral = AverageMeter()
    PRINT_FREQ = 100

    start = time.time()
    end = time.time()
    for idx, (samples, targets,samples_specral) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        samples_specral=samples_specral.cuda(non_blocking=True)
        output=model(samples,samples_specral)
        loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = util.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()
        lr_scheduler.step_update(epoch * num_steps + idx)

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_total_norm)
        batch_time.update(time.time() - end)
        end = time.time()
        acc_meter.update(accuracy(output, targets, topk=(1,))[0])

        # acc_meter_spectral.update(accuracy(output_spectral, targets, topk=(1,))[0])
        if (idx + 1) % PRINT_FREQ == 0 or (idx + 1) == num_steps:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{total_epochs}][{idx + 1}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                # f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'acc {acc_meter.val:.4f} ({acc_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    # print(f'train spectral accuracy:{acc_meter_spectral.val}')
    return loss_meter



@torch.no_grad()
def evaluate(
        model: torch.nn.Module,
        data_loader: DataLoader,
        criterion: torch.nn.Module ,
        acc, A, k, exp_time,
        logger=false_logger,
         ):
    model.eval()
    criterion.eval()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    # acc_meter_spectral = AverageMeter()
    PRINT_FREQ = 2000

    test_pred_all = []
    test_all = []
    correct = 0
    total = 0

    for idx, (samples, targets,samples_specral) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        samples_specral = samples_specral.cuda(non_blocking=True)
        output = model(samples, samples_specral)
        loss = criterion(output, targets)
        _, predicted = torch.max(output.data, 1)
        test_all = np.concatenate([test_all, targets.data.cpu().numpy()])
        test_pred_all = np.concatenate([test_pred_all, predicted.cpu()])
        correct += predicted.eq(targets.data.view_as(predicted)).cpu().sum()

        acc2 = accuracy(output, targets, topk=(1,))[0]
        end = time.time()
        loss_meter.update(loss.item(), targets.size(0))
        acc_meter.update(acc2.item(), targets.size(0))
        batch_time.update(time.time() - end)

        if (idx + 1) % PRINT_FREQ == 0 or (idx + 1) == num_steps:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'EVAL: [{(idx + 1)}/{len(data_loader)}]\t'
                # f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc {acc_meter.val:.3f} ({acc_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    acc[exp_time-1] = 100. * correct / len(data_loader.dataset)
    OA = acc
    C = metrics.confusion_matrix(test_all, test_pred_all)
    A[exp_time-1, :] = np.diag(C) / np.sum(C, 1, dtype=np.float)

    k[exp_time-1] = metrics.cohen_kappa_score(test_all, test_pred_all)

    logger.info(f'EVAL * Acc@ {acc_meter.avg:.3f}')
    return acc_meter.avg, loss_meter.avg, acc, A, k
