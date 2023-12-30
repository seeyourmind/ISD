#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File   :   distill_script.py    
@Time   :   2021/5/5 16:23
@Author :   Fyzer
@Description:   
"""
import os
import time
import copy
import numpy as np
import scipy.io as sio
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torchvision.models as models
from configs import parser
from solver.train import *
from solver.image_dataset import DistillDataSet
import torch.utils.data as data
from solver.utils import fix_seeds, config_process, split_class_ps, split_class_ss, get_whole_data
from backbone.ZSCRF import ZeroShotCRF


class DistillKL(nn.Module):
    """KL divergence for distillation"""
    def __init__(self, T=4, weight=None):
        super(DistillKL, self).__init__()
        self.T = T
        self.weight = weight

    def forward(self, y_s, y_t):
        if self.weight is not None:
            if self.weight.is_cuda:
                y_s = torch.mm(y_s, self.weight.t())
            else:
                y_s = y_s @ self.weight.t().cuda()
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss


class ContrastiveLoss(nn.Module):
    """contrastive loss for self-supervision of unseen"""
    def __init__(self, negative):
        super(ContrastiveLoss, self).__init__()
        self.negative = negative

    def forward(self, x, positive):
        bn, _ = x.size()
        cn, _ = self.negative.size()
        x = F.normalize(x)
        positive = F.normalize(positive)
        sim = F.cosine_similarity(x, positive, dim=1)
        x = x.unsqueeze(1).repeat(1, cn, 1)
        negative = self.negative.unsqueeze(0).repeat(bn, 1, 1)
        if negative.is_cuda:
            p = sim.exp() / torch.sum(F.cosine_similarity(x, negative, dim=2).exp(), dim=1)
        else:
            negative = negative.cuda()
            p = sim.exp() / torch.sum(F.cosine_similarity(x, negative, dim=2).exp(), dim=1)
        loss = -torch.log(p).sum()/bn
        # print(f'contrastive loss: {loss}')
        # exit()
        return loss


class SimpleContrastiveLoss(nn.Module):
    """contrastive loss implementation of invariance-equivariance"""
    def __init__(self, negative, temp=0.1):
        super(SimpleContrastiveLoss, self).__init__()
        self.negative = negative
        self.temp = temp

    def forward(self, x, positive):
        # Define constant eps to ensure training is not impacted if norm of any image rep is zero
        eps = 1e-6
        # L2 normalize x, positive and negative representations
        norm_x = torch.norm(x, dim=1, keepdim=True)
        norm_pos = torch.norm(positive, dim=1, keepdim=True)
        norm_neg = torch.norm(self.negative, dim=1, keepdim=True)
        x = (x / (norm_x + eps)).float()
        pos = (positive / (norm_pos + eps)).float()
        neg = (self.negative / (norm_neg + eps)).float()
        # Find cosine similarities
        sim_pos = (x @ pos.t()).diagonal()
        sim_neg = (x @ neg.t())
        # Fine exponentiation of similarity arrays
        exp_sim_pos = torch.exp(sim_pos / self.temp)
        exp_sim_neg = torch.exp(sim_neg / self.temp)
        # Sum exponential similarities of I_t with different images from memory bank of negatives
        sum_exp_sim_neg = torch.sum(exp_sim_neg, 1)

        # Find batch probabilities arr
        batch_prob = exp_sim_pos / (sum_exp_sim_neg + eps)

        neg_log_img_pair_probs = -1 * torch.log(batch_prob)
        loss = torch.sum(neg_log_img_pair_probs) / neg_log_img_pair_probs.size()[0]
        return loss


"""def get_distill_loader(cfg, split='PS', shuffle=True, drop_last=True):
    all_data = sio.loadmat(f'mat/{split}/syn_{cfg["dataset"]}.mat')
    re_inp = all_data['input']
    re_tar = all_data['target']
    re_out = all_data['output']
    re_att = all_data['attribute']
    if split == 'PS':
        att_splits = sio.loadmat(cfg['att_splits_ps'])
    elif split == 'SS':
        att_splits = sio.loadmat(cfg['att_splits_ss'])
    else:
        print('input invalid [split] (only PS or SS).')
    test_unseen_loc = att_splits['test_unseen_loc'] - 1
    original_att = att_splits['original_att'].T
    reinp = all_data['image_files'][test_unseen_loc].squeeze()
    retar = all_data['synthesis'][test_unseen_loc, :].squeeze()
    labels = all_data['labels'].squeeze()[test_unseen_loc].squeeze()

    unseen_classes = np.unique(labels)
    seen_classes = list(filter(lambda x: x not in unseen_classes, range(len(original_att))))
    unseen_att = original_att[unseen_classes, :]
    seen_att = original_att[seen_classes, :]

    labels = np.array(list(map(lambda x: np.where(unseen_classes == x)[0], labels))).squeeze()

    params = {'batch_size': config['batch_size'],
              'num_workers': 8,
              'pin_memory': True,
              'shuffle': shuffle,
              'drop_last': drop_last,
              'sampler': None}
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tr_transforms, ts_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, (0.88, 1), (0.5, 4.0 / 3)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]), transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    train_loader = data.DataLoader(DistillTrainDataSet(cfg, reinp, retar, tr_transforms), **params)
    params['shuffle'] = False
    test_loader = data.DataLoader(DistillTestDataSet(config, reinp, labels, ts_transforms), **params)
    return train_loader, test_loader, torch.from_numpy(seen_att).float(), torch.from_numpy(unseen_att).float()"""
def get_distill_loader(cfg, split='PS', shuffle=True, drop_last=True):
    all_data = torch.load(f'mat/{split}/syn_{cfg["dataset"]}.pt')
    re_inp = all_data['input']
    re_tar = all_data['target'].squeeze()
    re_out = all_data['output']
    re_att = all_data['attribute']
    trw = all_data['train_weight']
    tew = all_data['test_weight']

    params = {'batch_size': config['batch_size'],
              'num_workers': 8,
              'pin_memory': True,
              'shuffle': shuffle,
              'drop_last': drop_last,
              'sampler': None}

    train_loader = data.DataLoader(DistillDataSet(re_inp, re_out), **params)
    params['shuffle'] = False
    test_loader = data.DataLoader(DistillDataSet(re_inp, re_tar), **params)
    return train_loader, test_loader, trw, tew


def get_generalize_distill_loader(cfg, shuffle=True, drop_last=True):
    all_data = torch.load(f'mat/gen/syn_{cfg["dataset"]}.pt')
    re_inp = all_data['input']
    re_tar = all_data['target'].squeeze()
    re_out = all_data['output']
    attw = all_data['att_weight']
    trw = all_data['train_weight']
    tew = all_data['test_weight']

    params = {'batch_size': config['batch_size'],
              'num_workers': 8,
              'pin_memory': True,
              'shuffle': shuffle,
              'drop_last': drop_last,
              'sampler': None}

    train_loader = data.DataLoader(DistillDataSet(re_inp, re_out), **params)
    params['shuffle'] = False
    test_loader = data.DataLoader(DistillDataSet(re_inp, re_tar), **params)
    return train_loader, test_loader, attw, trw, tew


def test(config, model, criterion, test_loader, writer):
    # print('......train student......')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    if config['split'] == 'gen':
        class_avg = ClassAverageMeter(config['n_classes'])
    else:
        class_avg = ClassAverageMeter(config['n_test_lbl'])
    # freeze BatchNormalization and Dropout
    model.eval()
    print('......validate......')
    with torch.no_grad():
        end = time.time()
        for i, (inputs, target) in enumerate(test_loader):
            data_time.update(time.time() - end)
            inputs = inputs.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            # print(f'{target.dtype}')
            # exit()

            if config['split'] == 'gen':
                _, output = model(inputs, False)
                output = output @ config['att_weight'].t().cuda()
            else:
                output, _ = model(inputs, False)
            m = inputs.size(0)

            loss = criterion(output, target)
            avg_output = output

            # measure accuracy and record loss
            if config['split'] == 'gen':
                prec1, class_acc, class_cnt, pred_prob = accuracy(avg_output, target, config['n_classes'])
            else:
                prec1, class_acc, class_cnt, pred_prob = accuracy(avg_output, target, config['n_test_lbl'])
            losses.update(loss, m)
            top1.update(prec1, m)
            class_avg.update(class_acc, class_cnt, pred_prob)

            # time measure
            batch_time.update(time.time() - end)
            end = time.time()
            if writer is not None:
                writer.add_scalars('Loss', {'Test': loss.item()}, 0)
                writer.add_scalars('Prec@1', {'Test': top1.avg}, 0)
            if i % config['print_freq'] == 0:
                print('Test: [{0}/{1}] '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                      'Loss {loss.val:.4f} (avg: {loss.avg:.4f}) '
                      'Prec@1 {top1.val:.3f} (avg: {top1.avg:.3f}) '
                      'Class avg {class_avg.avg:.3f} '.format(i, len(test_loader), batch_time=batch_time,
                                                              loss=losses, class_avg=class_avg, top1=top1))
            if i >= 120:
                break

    return class_avg.avg, top1.avg, losses.avg


def train(config, model, optimizer, criterion, train_loader, epoch, lr_period, start_batch_idx, writer, is_seen=False):
    print('......train student......')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    # # 过拟合单batch
    for i, (inputs, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        inputs = inputs.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        if config['lr_strategy'] == 'sgdr_lr':
            set_optimizer_lr(config['lr']*sgdr(lr_period, i+start_batch_idx), optimizer)

        if config['split'] == 'gen':
            _, output = model(inputs, is_seen)
        else:
            output, _ = model(inputs, is_seen)
        m = inputs.size(0)

        loss = criterion(output, target)
        # loss = criterion[0](output, target) + criterion[1](attr, att)
        losses.update(loss, m)

        # gradient and SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # time measure
        batch_time.update(time.time() - end)
        end = time.time()
        if writer is not None:
            writer.add_scalars('Loss', {'Train': loss.item()}, 0)
        if i % config['print_freq'] == 0:
            print('Epoch: [{0}][{1}/{2}] '
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Loss {loss.val:.4f} (avg: {loss.avg:.4f}) '.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses))


def distill(config, model, optimizer, criterion, train_loader, test_loader):
    # writer = SummaryWriter()
    writer = None
    if config['pretrain'] == 0:
        best_epoch = -1
        best_meas = -1
        best_pred_prob = None

    if config['lr_strategy'] == 'sgdr_lr':
        lr_period = config['cycle_len'] * len(train_loader)
    else:
        lr_period = 0

    for epoch in range(config['start_epoch'], config['epochs']):

        start_batch_idx = len(train_loader) * epoch

        train(config, model, optimizer, criterion[1], train_loader, epoch, lr_period, start_batch_idx, writer=writer)
        test_acc, test_top1, pred_prob = test(config, model, criterion[0], test_loader, writer=writer)

        is_best = test_acc > best_meas
        if is_best:
            best_epoch = epoch
            best_meas = test_acc
            best_pred_prob = pred_prob
            save_model(config, model, optimizer, epoch, best_meas, best_epoch, True, './models/Distill-{}-{}-{:.3f}{}'.format(config['dataset'], config['split'], best_meas, '_checkpoint.pth.tar'))
        config['iter'] = epoch
        config['test_acc'] = test_acc
        config['test_top1'] = test_top1
        config['best_epoch'] = best_epoch
        config['best_meas'] = best_meas

        print('[current]\tepochs {} train {:.3f} pred meas {:.3f}\t[best]\tepochs {} test {:.3f} pred meas {:.3f}'.format(epoch, test_top1, test_acc, best_epoch, best_meas, test_acc))
        if epoch - best_epoch >= 20:
            print("Early stopping")
            break
    return best_meas, best_epoch, best_pred_prob


def infer(best_cfg, logger, split='PS'):
    best_cfg['split'] = split
    print('......generate train set and test set.......')
    # data grab
    if split == 'gen':
        train_loader, test_loader, att_weight, seen_weight, unseen_weight = get_generalize_distill_loader(best_cfg)
        best_cfg['att_weight'] = att_weight
    else:
        train_loader, test_loader, seen_weight, unseen_weight = get_distill_loader(best_cfg, split)

    best_cfg['n_classes'] = seen_weight.size(0) + unseen_weight.size(0)
    best_cfg['n_train_lbl'] = seen_weight.size(0)
    best_cfg['n_test_lbl'] = unseen_weight.size(0)

    train_linear_weight = F.normalize(seen_weight)
    test_linear_weight = F.normalize(unseen_weight)

    print('......create model.......')
    model = models.__dict__['resnet101'](pretrained=True)
    model = ZeroShotCRF(best_cfg, model, Parameter(train_linear_weight), Parameter(test_linear_weight))
    # model = ZeroShotCRF(best_cfg, model, Parameter(datasets[0][5]), Parameter(datasets[0][6]))
    model = nn.DataParallel(model).cuda()  # multi GUP acceleration

    best_cfg['pre_model'] = '/mnt/samsung/fangzhiyu/VULCAN_Python/ZSCRF-distill/models/Distill-CUB-SS-67.177_checkpoint.pth.tar'
    load_model(best_cfg, model, None, best_cfg['pre_model'])
    # define loss and optimizer
    if split == 'gen':
        criterion = [nn.CrossEntropyLoss().cuda(), DistillKL(weight=att_weight).cuda()]
    else:
        criterion = [nn.CrossEntropyLoss().cuda(), DistillKL().cuda()]
    optimizer = torch.optim.SGD(model.parameters(), lr=best_cfg['lr'], momentum=best_cfg['momentum'], weight_decay=best_cfg['weight_decay'])
    # optimizer = SWA(optimizer, swa_start=10, swa_freq=5, swa_lr=0.05)

    # train model
    best_cfg['pretrain'] = 0
    best_cfg['train_weight'] = train_linear_weight
    best_cfg['test_weight'] = test_linear_weight

    best_meas, best_epoch, best_pred_prob = distill(best_cfg, model, optimizer, criterion, train_loader, test_loader)
    print('Reproducing {}:{} ACA = {:.3f}% epoch = {}\n\n'.format(best_cfg['dataset'], split, best_meas, best_epoch))
    logger.info('ACA={:.3f}% cycle_len={} parts={} dropout={} threshold={} lr={} momentum={} weight_decay={} epoch={}'.
                format(best_meas, best_cfg['cycle_len'], best_cfg['parts'], best_cfg['dropout'], best_cfg['threshold'],
                       best_cfg['lr'], best_cfg['momentum'], best_cfg['weight_decay'], best_epoch))


if __name__ == '__main__':
    '''for reproducing purpose on CUB:PS ZSL results! '''
    torch.multiprocessing.set_start_method('spawn')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    print('execute main......')
    config = fix_seeds(config_process(parser.parse_args()))
    best_cfg = config
    best_cfg = fix_seeds(best_cfg)

    logger = get_logger('./results/distill_ps_{}.log'.format(time.strftime("%Y%m%d%H%M%S", time.localtime())))
    best_cfg['parts'] = 10
    best_cfg['lr'] = 0.0001
    best_cfg['cycle_len'] = 20
    best_cfg['dropout'] = 0.3
    best_cfg['threshold'] = 0.7
    infer(best_cfg, logger, 'SS')#76.212

