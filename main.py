#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import torch.nn as nn
import time
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from configs import parser
from solver.train import *
from solver.utils import fix_seeds, config_process, split_class_ps
from backbone.vision_transformer import VisionTransformer


def train(config, model, optimizer, criterion, train_loader, epoch, lr_period, start_batch_idx, writer):
    # print('begin train......')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    class_avg = ClassAverageMeter(config['n_train_lbl'])

    end = time.time()

    # 过拟合单batch
    first_batch = next(iter(train_loader))
    print(f'first batch: {first_batch.shape}')
    exit()
    for i, (inputs, target) in enumerate(train_loader):
        # switch to train mode
        model.train()
        data_time.update(time.time() - end)
        inputs = inputs.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        if config['lr_strategy'] == 'sgdr_lr':
            set_optimizer_lr(config['lr']*sgdr(lr_period, i+start_batch_idx), optimizer)

        # output = model(inputs)
        out, px, pxy = model(inputs)
        m = inputs.size(0)

        # print('loss = ', torch.sum(mi)/m)
        # compute loss value and output value
        ls_lmbd3 = config['ls_coef_bi']
        ls_lmbd2 = config['ls_coef_part']
        lmbd3 = config['coef_bi']
        lmbd2 = config['coef_part']
        # loss = (ls_lmbd2*criterion(output[0], target)+ls_lmbd3*criterion(output[1], target))/(ls_lmbd2+ls_lmbd3)
        loss = criterion[0](out, target) + criterion[1](pxy, px)
        # loss = criterion[0](out[0], target) + (torch.sum(out[1] * criterion[1](out[2], out[1])))/m
        avg_output = out
        # avg_output = (lmbd2 * output[0] + lmbd3 * output[1]) / (lmbd2 + lmbd3)

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
    # optimizer.swap_swa_sgd()

    return class_avg.avg, top1.avg, losses.avg


def test(config, model, criterion, test_loader, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    class_avg = ClassAverageMeter(config['n_test_lbl'])

    # freeze BatchNormalization and Dropout
    model.eval()

    with torch.no_grad():
        end = time.time()
        # first_batch = next(iter(val_loader))
        for i, (inputs, target) in enumerate(test_loader):
            data_time.update(time.time() - end)

            inputs = inputs.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = model(inputs)
            # out = model(inputs)
            m = inputs.size(0)

            # compute loss value and output value
            ls_lmbd3 = config['ls_coef_bi']
            ls_lmbd2 = config['ls_coef_part']
            lmbd3 = config['coef_bi']
            lmbd2 = config['coef_part']
            loss = (ls_lmbd2 * criterion(output[0], target) + ls_lmbd3 * criterion(output[1], target)) / (
                        ls_lmbd2 + ls_lmbd3)
            # loss = criterion(out, target) + compute_mi(mi, 0.3)/m
            # loss = criterion[0](out[0], target) + torch.sum(out[1] * criterion[1](out[2], out[1])) / m
            # avg_output = out
            avg_output = (lmbd2 * output[0] + lmbd3 * output[1]) / (lmbd2 + lmbd3)

            # measure accuracy and record loss
            prec1, class_acc, class_cnt, pred_prob = accuracy(avg_output, target, config['n_test_lbl'])
            losses.update(loss, m)
            top1.update(prec1, m)
            class_avg.update(class_acc, class_cnt, pred_prob)

            # time measure
            batch_time.update(time.time() - end)
            end = time.time()
            writer.add_scalars('Loss', {'Test': loss.item()}, 0)
            writer.add_scalars('Prec@1', {'Test': top1.avg}, 0)
            if i % config['print_freq'] == 0:
                print('Test: [{0}/{1}] '
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Loss {loss.val:.4f} (avg: {loss.avg:.4f}) '
                            'Prec@1 {top1.val:.3f} (avg: {top1.avg:.3f}) '
                            'Class avg {class_avg.avg:.3f} '.format(i, len(test_loader), batch_time=batch_time, loss=losses, class_avg=class_avg, top1=top1))

    return class_avg.avg, top1.avg, losses.avg


def valid(config, model, criterion, val_loader, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    class_avg = ClassAverageMeter(config['n_train_lbl'])

    # freeze BatchNormalization and Dropout
    model.eval()

    with torch.no_grad():
        end = time.time()
        # first_batch = next(iter(val_loader))
        for i, (inputs, target) in enumerate(val_loader):
            data_time.update(time.time() - end)

            inputs = inputs.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = model(inputs)
            # out = model(inputs)
            m = inputs.size(0)

            # compute loss value and output value
            ls_lmbd3 = config['ls_coef_bi']
            ls_lmbd2 = config['ls_coef_part']
            lmbd3 = config['coef_bi']
            lmbd2 = config['coef_part']
            loss = (ls_lmbd2 * criterion(output[0], target) + ls_lmbd3 * criterion(output[1], target)) / (
                    ls_lmbd2 + ls_lmbd3)
            # loss = criterion(out, target) + compute_mi(mi, 0.3)/m
            # loss = criterion[0](out[0], target) + torch.sum(out[1] * criterion[1](out[2], out[1])) / m
            # avg_output = out
            avg_output = (lmbd2 * output[0] + lmbd3 * output[1]) / (lmbd2 + lmbd3)

            # measure accuracy and record loss
            prec1, class_acc, class_cnt, pred_prob = accuracy(avg_output, target, config['n_train_lbl'])
            losses.update(loss, m)
            top1.update(prec1, m)
            class_avg.update(class_acc, class_cnt, pred_prob)

            # time measure
            batch_time.update(time.time() - end)
            end = time.time()
            writer.add_scalars('Loss', {'Valid': loss.item()}, 0)
            writer.add_scalars('Prec@1', {'Valid': top1.avg}, 0)
            if i % config['print_freq'] == 0:
                print('Valid: [{0}/{1}] '
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Loss {loss.val:.4f} (avg: {loss.avg:.4f}) '
                            'Prec@1 {top1.val:.3f} (avg: {top1.avg:.3f}) '
                            'Class avg {class_avg.avg:.3f} '.format(i, len(val_loader), batch_time=batch_time, loss=losses, class_avg=class_avg, top1=top1))

    return class_avg.avg, top1.avg, losses.avg


def train_test(config, model, optimizer, criterion, train_loader, test_loader):
    writer = SummaryWriter()
    if config['pretrain'] == 0:
        best_epoch = -1
        best_meas = -1
        best_pred_prob = None

    if config['lr_strategy'] == 'sgdr_lr':
        lr_period = config['cycle_len'] * len(train_loader)

    for epoch in range(config['start_epoch'], config['epochs']):

        start_batch_idx = len(train_loader) * epoch

        train_acc, train_top1, train_loss = train(config, model, optimizer, criterion, train_loader, epoch, lr_period, start_batch_idx, writer=writer)
        test_acc, test_top1, pred_prob = test(config, model, criterion, test_loader, writer=writer)

        is_best = test_acc > best_meas
        if is_best:
            best_epoch = epoch
            best_meas = test_acc
            best_pred_prob = pred_prob
            save_model(config, model, optimizer, epoch, best_meas, best_epoch, True, './models/{}{}'.format(config['output'], '_checkpoint.pth.tar'))
        config['iter'] = epoch
        config['train_acc'] = train_acc
        config['train_top1'] = train_top1
        config['test_acc'] = test_acc
        config['test_top1'] = test_top1
        config['train_loss'] = train_loss.item()
        config['best_epoch'] = best_epoch
        config['best_meas'] = best_meas

        print('[current]\tepochs {} train {:.3f} pred meas {:.3f}\t[best]\tepochs {} test {:.3f} pred meas {:.3f}'.format(epoch, train_top1, train_acc, best_epoch, best_meas, test_acc))
    return best_meas, best_epoch, best_pred_prob


def infer(best_cfg, datasets):
    # logger = get_logger('./results/train_{}.log'.format(time.strftime("%Y%m%d%H%M%S", time.localtime())))
    # attr_w2v = torch.from_numpy(np.load('./models/attr_mat_w2v.npy').astype(np.float32)).cuda()
    for i in [0.5]:
        best_cfg['dropout'] = i
        print('inference classify stream.......')
        # data grab
        train_set = grab_data(best_cfg, datasets[0][0], datasets[0][3], attr=datasets[0][4], is_train=True)
        test_set = grab_data(best_cfg, datasets[0][1], datasets[0][3], attr=datasets[0][4], is_train=False)
        # val_set = grab_data(best_cfg, datasets[0][2], datasets[0][3], attr=None, is_train=False)
        print('generate train set and test set.......')
        model = models.__dict__['resnet101'](pretrained=True)
        model = nn.Sequential(*list(model.children())[:-2])
        print(f"embed_dim: {best_cfg['attr_dim']}")
        model = VisionTransformer(img_size=224, embed_dim=best_cfg['attr_dim'], num_heads=26, no_head=True, hybrid_backbone=model)
        # model = Baseline(best_cfg, model=model, train_attr=Parameter(F.normalize(datasets[0][4])), test_attr=Parameter(F.normalize(datasets[0][5])))
        model = nn.DataParallel(model).cuda()  # multi GUP acceleration
        print('create model.......')

        # define loss and optimizer
        criterion = [nn.CrossEntropyLoss().cuda(), nn.KLDivLoss(reduction='mean').cuda()]
        optimizer = torch.optim.SGD(model.parameters(), lr=best_cfg['lr'], momentum=best_cfg['momentum'], weight_decay=best_cfg['weight_decay'])
        # optimizer = SWA(optimizer, swa_start=10, swa_freq=5, swa_lr=0.05)

        # train model
        best_cfg['ls_coef_bi'] = 1
        best_cfg['coef_bi'] = 1
        best_cfg['ls_coef_part'] = 0
        best_cfg['coef_part'] = 0
        best_cfg['pretrain'] = 0
        best_cfg['lr_strategy'] = 'sgdr_lr'
        print(f"start training!  dropout[{i}]")
        best_meas, best_epoch, best_pred_prob = train_test(best_cfg, model, optimizer, criterion, train_set, test_set)
        print('[Dropout {}] Reproducing CUB:PS ACA = {:.3f}% epoch = {}\n\n'.format(i, best_meas, best_epoch))


if __name__ == '__main__':
    '''for reproducing purpose on CUB:PS ZSL results! '''
    torch.multiprocessing.set_start_method('spawn')
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    print('execute main......')
    config = fix_seeds(config_process(parser.parse_args()))
    # # train, test, label, train_attr, test_attr
    datasets = split_class_ps(config)
    # examples, labels, class_map = image_load(config['class_file'], config['image_label'])
    # datasets = split_byclass(config, examples, labels, np.loadtxt(config['attributes_file']), class_map)
    print(f'load train set {len(datasets[0][0])} test set {len(datasets[0][1])}')
    best_cfg = config
    best_cfg = fix_seeds(best_cfg)
    best_cfg['epochs'] = config['final_epoch']
    best_cfg['parts'] = 10
    best_cfg['n_classes'] = datasets[0][5].size(0) + datasets[0][6].size(0)
    best_cfg['n_train_lbl'] = datasets[0][5].size(0)
    best_cfg['n_test_lbl'] = datasets[0][6].size(0)

    infer(best_cfg, datasets)
