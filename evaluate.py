#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import time
import scipy.io as sio
from tqdm import tqdm
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torchvision.models as models
from configs import parser
from solver.train import *
from solver.utils import fix_seeds, config_process, get_whole_data
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


def infer_test(best_cfg):
    print('inference classify stream.......')
    # data grab
    datasets = get_whole_data(config)
    exit()
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
                        infer_test(best_cfg)
