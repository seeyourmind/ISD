#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from solver.image_dataset import DataSet, Data2Set
import shutil
import math
import logging
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ClassAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, n_labels):
        self.reset(n_labels)

    def reset(self, n_labels):
        self.n_labels = n_labels
        self.acc = torch.zeros(n_labels)
        self.cnt = torch.Tensor([1e-8] * n_labels)
        self.pred_prob = []

    def update(self, val, cnt, pred_prob):
        self.acc += val
        self.cnt += cnt
        self.avg = 100 * self.acc.dot(1.0 / self.cnt).item() / self.n_labels
        self.pred_prob += pred_prob
        # print ('pred',len(self.pred_prob))`


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def grab_data(config, examples, labels, attr=None, is_train=True, drop_last=True):
    params = {'batch_size': config['batch_size'],
              'num_workers': 8,
              'pin_memory': True,
              'drop_last': drop_last,
              'sampler': None}
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tr_transforms, ts_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, (0.88, 1), (0.5, 4.0/3)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]), transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    if is_train:
        params['shuffle'] = True
        params['sampler'] = None
        data_set = data.DataLoader(DataSet(config, examples, labels, attr, tr_transforms, is_train), **params)
    else:
        params['shuffle'] = False
        data_set = data.DataLoader(DataSet(config, examples, labels, attr, ts_transforms, is_train), **params)
    return data_set


def general_grab_data(config, test_seen, test_unseen, labels, attr=None, is_train=True, drop_last=True):
    params = {'batch_size': config['batch_size'],
              'num_workers': 8,
              'pin_memory': True,
              'drop_last': drop_last,
              'shuffle': False,
              'sampler': None}
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ts_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    seen_dty = np.ones(len(test_seen))
    unseen_dty = np.zeros(len(test_unseen))
    example_seen = np.concatenate((test_seen, seen_dty), axis=1)
    example_unseen = np.concatenate((test_unseen, unseen_dty), axis=1)
    examples = np.concatenate((example_seen, example_unseen), axis=0)
    data_set = data.DataLoader(Data2Set(config, examples, labels, attr, ts_transforms, is_train), **params)
    return data_set


def load_model(config, model, optimizer, fname):
    checkpoint = torch.load(fname)
    config['start_epoch'] = checkpoint['epoch']
    config['best_meas'] = checkpoint['best_meas']
    config['best_epoch'] = checkpoint['best_epoch']

    pretrained_dict = checkpoint['state_dict']
    model_dict = model.state_dict()
    # 1.filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2.overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3.load the new state dict
    model.load_state_dict(model_dict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch: {}), best_prec: {})".format(fname, checkpoint['epoch'], checkpoint['best_meas']))


def save_model(config, model, optimizer, epoch, best_meas, best_epoch, is_best, fname):
    state = {'epoch': epoch+1,
             'arch': 'resnet101',
             'state_dict': model.state_dict(),
             'best_meas': best_meas,
             'best_epoch': best_epoch,
             'optimizer': optimizer.state_dict()}
    torch.save(state, fname)
    if is_best:
        shutil.copyfile(fname, './models/{}_model_best.pth.tar'.format(config['output']))


def adjust_learning_rate(optimizer, epoch, config):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = config['lr']*(0.1 ** (epoch // config['step']))
    print('current step learning rate {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def set_optimizer_lr(lr, optimizer):
    # callback to set the learning rate in an optimizer, without rebuilding the whole optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# $$\frac{1}{2}(1+cos(\frac{T_{cur}}{T_{i}}\pi))$$
def sgdr(period, batch_idx):
    # SGDR learning rate
    '''returns normalised anytime sgdr schedule given period and batch_idx
        best performing settings reported in paper are T_0 = 10, T_mult=2
        so always use T_mult=2 ICLR-17 SGDR learning rate.'''
    batch_idx = float(batch_idx)
    restart_period = period
    while batch_idx / restart_period > 1.:
        batch_idx = batch_idx - restart_period
        restart_period = restart_period * 2.
    r = math.pi * (batch_idx / restart_period)
    return 0.5 * (1.0 + math.cos(r))


def accuracy(output_vec, target, n_labels):
    """Computes the precision@k for the specified values of k"""
    output = output_vec

    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    # print('pred: ', pred.view(-1))  # output the prediction for one batch
    class_accuracy = torch.zeros(n_labels)
    class_cnt = torch.zeros(n_labels)
    prec = 0.0
    pred_prob = []
    for i in range(batch_size):
        t = target[i]
        pred_prob.append(output[i][t])

        if pred[i] == t:
            prec += 1
            class_accuracy[t] += 1
        class_cnt[t] += 1
    return prec * 100.0 / batch_size, class_accuracy, class_cnt, pred_prob


def compute_mi(mi, mi_threshold):
    b = mi.size(0)
    # handle dirty data
    mi = torch.where(torch.isinf(mi), torch.full_like(mi, 1), mi)
    mi = torch.where(torch.isnan(mi), torch.full_like(mi, 0), mi)
    # normalize
    mi = (mi - mi.min()) / (mi.max() - mi.min())
    mi_selected = torch.where((0.0 < mi) & (mi <= mi_threshold), mi, torch.full_like(mi, 0.0))
    mi_batch = torch.sum(mi_selected.view(b, -1), dim=1)
    print(f'mi_selected min {mi_selected.min()} max {mi_selected.max()} sum {torch.sum(mi_batch)}')
    return torch.sum(mi_batch)
