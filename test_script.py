#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import time
import copy
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from configs import parser
from solver.train import *
from solver.image_dataset import DistillDataSet
import torch.utils.data as data
from solver.utils import fix_seeds, config_process, split_class_ps, split_class_ss, get_whole_data
from backbone.ZSCRF import ZeroShotCRF


class DistillKL(nn.Module):
    """KL divergence for distillation"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
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
        sim = F.cosine_similarity(x, positive, dim=1)
        x = x.unsqueeze(1).repeat(1, cn, 1)
        negative = self.negative.unsqueeze(0).repeat(bn, 1, 1)
        if x.is_cuda:
            negative = negative.cuda()
        neg_sim = F.cosine_similarity(x, negative, dim=2).exp()
        p = sim.exp() / torch.sum(F.cosine_similarity(x, negative, dim=2).exp(), dim=1)
        loss = -torch.log(p).sum()/bn
        # print(f'contrastive loss: {loss}')
        # exit()
        return loss


def train_teacher(config, model, optimizer, criterion, train_loader, epoch, lr_period, start_batch_idx, writer, is_seen=True):
    print('......train teacher......')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    class_avg = ClassAverageMeter(config['n_train_lbl'])

    end = time.time()

    # # 过拟合单batch
    for i, (inputs, target) in enumerate(train_loader):
        # switch to train mode
        model.train()
        data_time.update(time.time() - end)
        inputs = inputs.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        if config['lr_strategy'] == 'sgdr_lr':
            set_optimizer_lr(config['lr']*sgdr(lr_period, i+start_batch_idx), optimizer)

        output = model(inputs, is_seen)
        m = inputs.size(0)

        loss = criterion(output, target)
        avg_output = output

        # measure accuracy and record loss
        prec1, class_acc, class_cnt, pred_prob = accuracy(avg_output, target, config['n_train_lbl'])
        losses.update(loss, m)
        top1.update(prec1, m)

        class_avg.update(class_acc, class_cnt, pred_prob)

        # gradient and SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # time measure
        batch_time.update(time.time() - end)
        end = time.time()
        if writer is not None:
            writer.add_scalars('Loss', {'Train': loss.item()}, 0)
            writer.add_scalars('Prec@1', {'Train': top1.avg}, 0)
        if i % config['print_freq'] == 0:
            print('Epoch: [{0}][{1}/{2}] '
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Loss {loss.val:.4f} (avg: {loss.avg:.4f}) '
                        'Prec@1 {top1.val:.3f} (avg: {top1.avg:.3f}) '
                        'Class avg {lbl_avg.avg:.3f} '.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, lbl_avg=class_avg, top1=top1))
    return class_avg.avg, top1.avg, losses.avg


def get_unseen_data_loader(model, test_loader):
    # generate unseen label
    # freeze BatchNormalization and Dropout
    model.eval()
    with torch.no_grad():
        reinp = []
        retar = []
        reatt = []
        for i, (inputs, target, att) in enumerate(test_loader):
            inputs = inputs.cuda(non_blocking=True)
            att = att.cuda(non_blocking=True)
            output,_ = model(inputs, False)
            reinp.append(inputs)
            retar.append(output)
            reatt.append(att)
    reinp = torch.cat(reinp).cpu().numpy()
    retar = torch.cat(retar).cpu().numpy()
    reatt = torch.cat(reatt).cpu().numpy()

    retest_loader = data.DataLoader(DistillDataSet(reinp, retar, att=reatt), batch_size=config['batch_size'], shuffle=True, drop_last=True)
    return retest_loader


def test_unseen(config, model, criterion, test_loader, writer):
    # print('......train student......')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    class_avg = ClassAverageMeter(config['n_test_lbl'])
    # freeze BatchNormalization and Dropout
    model.eval()
    print('......validate......')
    with torch.no_grad():
        end = time.time()
        for i, (inputs, target, _) in enumerate(test_loader):
            data_time.update(time.time() - end)
            inputs = inputs.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output,_ = model(inputs, False)
            m = inputs.size(0)

            loss = criterion(output, target)
            avg_output = output

            # measure accuracy and record loss
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

    return class_avg.avg, top1.avg, losses.avg


def train_student(config, model, optimizer, criterion, re_test, epoch, lr_period, start_batch_idx, writer, is_seen=False):
    print('......train student......')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    print('......distill......')
    # # 过拟合单batch
    for i, (inputs, target, att) in enumerate(re_test):
        data_time.update(time.time() - end)
        inputs = inputs.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        att = att.cuda(non_blocking=True)

        if config['lr_strategy'] == 'sgdr_lr':
            set_optimizer_lr(config['lr'] * sgdr(lr_period, i + start_batch_idx), optimizer)

        output, outatt = model(inputs, is_seen)
        m = inputs.size(0)
        output = F.softmax(output, dim=1)
        target = F.softmax(target, dim=1)
        loss = criterion[0](output.log(), target) + criterion[1](outatt, att)

        # gradient and SGD step
        losses.update(loss, m)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # time measure
        batch_time.update(time.time() - end)
        end = time.time()
        if writer is not None:
            writer.add_scalars('Loss', {'Train': loss.item()}, 0)

        if i % config['print_freq'] == 0:
            print('Train: [{0}/{1}] '
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss.val:.4f} (avg: {loss.avg:.4f}).'.format(i, len(re_test), batch_time=batch_time, loss=losses))
    return losses.val


def generate(model, test_loader, marker=''):
    # freeze BatchNormalization and Dropout
    model.eval()

    with torch.no_grad():
        for i, (inputs, target) in tqdm(enumerate(test_loader)):
            inputs = inputs.cuda(non_blocking=True)
            output = model(inputs, False)
            if i == 0:
                feat_mat = output
            else:
                feat_mat = torch.cat([feat_mat, output], dim=0)
    print(f'feat mat shape {feat_mat.shape}')
    feat_mat = feat_mat.cpu().numpy()
    sio.savemat(f'all_data{marker}.mat', {'features': feat_mat})


def distill(config, teacher, student, optimizer, criterion, train_loader, test_loader):
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

        re_test = get_unseen_data_loader(config, teacher, test_loader)
        train_student(config, student, optimizer, [criterion[1], criterion[2]], re_test, epoch, lr_period, start_batch_idx, writer=writer)
        test_acc, test_top1, pred_prob = test_unseen(config, student, criterion[0], test_loader, writer=writer)

        is_best = test_acc > best_meas
        if is_best:
            best_epoch = epoch
            best_meas = test_acc
            best_pred_prob = pred_prob
            save_model(config, student, optimizer, epoch, best_meas, best_epoch, True, './models/{}_{:.3f}{}'.format(config['output'], best_meas, '_checkpoint.pth.tar'))
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


def infer_test(best_cfg):
    print('inference classify stream.......')
    # data grab
    datasets = get_whole_data(config)
    print(f'load all set {len(datasets[0][0])}')
    best_cfg['n_classes'] = datasets[0][3].size(0) + datasets[0][4].size(0)
    best_cfg['n_train_lbl'] = datasets[0][3].size(0)
    best_cfg['n_test_lbl'] = datasets[0][4].size(0)
    train_linear_weight = F.normalize(datasets[0][3])
    test_linear_weight = F.normalize(datasets[0][4])
    all_set = grab_data(best_cfg, datasets[0][0], datasets[0][1], attr=None, is_train=False, drop_last=False)
    print('generate train set and test set.......')
    model = models.__dict__['resnet101'](pretrained=True)
    model = ZeroShotCRF(best_cfg, model, Parameter(train_linear_weight), Parameter(test_linear_weight))
    model = nn.DataParallel(model).cuda()  # multi GUP acceleration
    print('create model.......')

    # train model
    best_cfg['pretrain'] = 0
    best_cfg['train_weight'] = train_linear_weight
    best_cfg['test_weight'] = test_linear_weight

    load_model(best_cfg, model, None, best_cfg['pre_model'])
    generate(model, all_set, marker='_69.346')


def infer_distill(best_cfg, logger, split='PS'):
    print('......generate train set and test set.......')
    # data grab
    if split == 'PS':
        datasets = split_class_ps(best_cfg)
        print(f'load train set {len(datasets[0][0])} test set {len(datasets[0][1])}')
        best_cfg['n_classes'] = datasets[0][5].size(0) + datasets[0][6].size(0)
        best_cfg['n_train_lbl'] = datasets[0][5].size(0)
        best_cfg['n_test_lbl'] = datasets[0][6].size(0)

        train_linear_weight = F.normalize(datasets[0][5])
        test_linear_weight = F.normalize(datasets[0][6])
        train_set = grab_data(best_cfg, datasets[0][0], datasets[0][3], attr=datasets[0][4], is_train=True)
        test_set = grab_data(best_cfg, datasets[0][1], datasets[0][3], attr=datasets[0][4], is_train=False)
    elif split == 'SS':
        datasets = split_class_ss(best_cfg)
        print(f'load train set {len(datasets[0][0])} test set {len(datasets[0][1])}')
        best_cfg['n_classes'] = datasets[0][4].size(0) + datasets[0][5].size(0)
        best_cfg['n_train_lbl'] = datasets[0][4].size(0)
        best_cfg['n_test_lbl'] = datasets[0][5].size(0)

        train_linear_weight = F.normalize(datasets[0][4])
        test_linear_weight = F.normalize(datasets[0][5])
        train_set = grab_data(best_cfg, datasets[0][0], datasets[0][2], attr=None, is_train=True)
        test_set = grab_data(best_cfg, datasets[0][1], datasets[0][2], attr=None, is_train=False)
    else:
        print('split model is invalid')
        exit()

    print('......create model.......')
    model = models.__dict__['resnet101'](pretrained=True)
    model = ZeroShotCRF(best_cfg, model, Parameter(train_linear_weight), Parameter(test_linear_weight))
    # model = ZeroShotCRF(best_cfg, model, Parameter(datasets[0][5]), Parameter(datasets[0][6]))
    model = nn.DataParallel(model).cuda()  # multi GUP acceleration
    student = copy.deepcopy(model)

    # define loss and optimizer
    criterion = [nn.CrossEntropyLoss().cuda(), DistillKL(4).cuda(), ContrastiveLoss(test_linear_weight).cuda()]
    optimizer = torch.optim.SGD(model.parameters(), lr=best_cfg['lr'], momentum=best_cfg['momentum'], weight_decay=best_cfg['weight_decay'])
    # optimizer = SWA(optimizer, swa_start=10, swa_freq=5, swa_lr=0.05)

    # train model
    best_cfg['pretrain'] = 0
    best_cfg['train_weight'] = train_linear_weight
    best_cfg['test_weight'] = test_linear_weight
    student_cfg = copy.deepcopy(best_cfg)
    load_model(best_cfg, model, None, best_cfg['pre_model'])
    # best_cfg['start_epoch'] = 0
    student_cfg['batch_size'] = 32
    best_meas, best_epoch, best_pred_prob = distill(student_cfg, model, student, optimizer, criterion, train_set, test_set)
    print('Reproducing {}:PS ACA = {:.3f}% epoch = {}\n\n'.format(best_cfg['dataset'], best_meas, best_epoch))
    logger.info('ACA={:.3f}% cycle_len={} parts={} dropout={} threshold={} lr={} momentum={} weight_decay={} epoch={}'.
                format(best_meas, best_cfg['cycle_len'], best_cfg['parts'], best_cfg['dropout'], best_cfg['threshold'],
                       best_cfg['lr'], best_cfg['momentum'], best_cfg['weight_decay'], best_epoch))



if __name__ == '__main__':
    '''for reproducing purpose on CUB:PS ZSL results! '''
    torch.multiprocessing.set_start_method('spawn')
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    print('execute main......')
    config = fix_seeds(config_process(parser.parse_args()))
    best_cfg = config
    best_cfg = fix_seeds(best_cfg)

    cycle_len = [10]
    parts = [10]
    dropout = [0.3]#[0.2, 0.2, 0.2]
    threshold = [0.7]#[0.7, 0.8]
    lr = [0.0001]
    logger = get_logger('./results/distill_IZSwSD_{}.log'.format(time.strftime("%Y%m%d%H%M%S", time.localtime())))
    for clidx in cycle_len:
        best_cfg['cycle_len'] = clidx
        for pidx in parts:
            best_cfg['parts'] = pidx
            for didx in dropout:
                best_cfg['dropout'] = didx
                for tidx in threshold:
                    best_cfg['threshold'] = tidx
                    for lridx in lr:
                        best_cfg['lr'] = lridx
                        # infer_test(best_cfg)
                        infer_distill(best_cfg, logger, 'PS')
