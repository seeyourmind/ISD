#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File   :   get_synthesis_label.py    
@Time   :   2021/5/5 13:58
@Author :   Fyzer
@Description:   
"""
import os
import scipy.io as sio
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torchvision.models as models
from configs import parser
from solver.train import *
from solver.utils import fix_seeds, config_process, split_class_ps, split_class_ss, get_generalize_data
from backbone.ZSCRF import ZeroShotCRF


def generate(cfg, model, test_loader, train_weight, test_weight, split='PS'):
    print(f'data set: {cfg["dataset"]}')
    # freeze BatchNormalization and Dropout
    model.eval()

    with torch.no_grad():
        inp_mat = []
        tar_mat = []
        out_mat = []
        att_mat = []
        for i, (inputs, target, att) in tqdm(enumerate(test_loader)):
            inputs = inputs.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            att = att.cuda(non_blocking=True)
            output, _ = model(inputs, False)

            inp_mat.append(inputs)
            tar_mat.append(target)
            out_mat.append(output)
            att_mat.append(att)

        inp_mat = torch.cat(inp_mat).cpu().clone()
        tar_mat = torch.cat(tar_mat).cpu().clone()
        out_mat = torch.cat(out_mat).cpu().clone()
        att_mat = torch.cat(att_mat).cpu().clone()
        train_weight = train_weight.clone()
        test_weight = test_weight.clone()
    print(f'imgfiles: {inp_mat.shape}, synthesis: {out_mat.shape}, labels: {tar_mat.shape}')
    mat_dict = {'input': inp_mat,
                'target': tar_mat,
                'attribute': att_mat,
                'output': out_mat,
                'train_weight': train_weight,
                'test_weight': test_weight}
    torch.save(mat_dict, f'mat/{split}/syn_{cfg["dataset"]}.pt')
    print(f'save file: mat/{split}/syn_{cfg["dataset"]}.pt')


def generalize(cfg, model, data_loader, att_weight, train_weight, test_weight, split='PS'):
    print(f'data set: {cfg["dataset"]}')
    # freeze BatchNormalization and Dropout
    model.eval()

    with torch.no_grad():
        inp_mat = []
        tar_mat = []
        out_mat = []
        print('inference on test seen data set')
        for i, (inputs, target) in tqdm(enumerate(data_loader)):
            inputs = inputs.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            att_weight = att_weight.cuda(non_blocking=True)
            _, output = model(inputs, False)
            output = output @ att_weight.t()

            inp_mat.append(inputs)
            tar_mat.append(target)
            out_mat.append(output)

        inp_mat = torch.cat(inp_mat).cpu().clone()
        tar_mat = torch.cat(tar_mat).cpu().clone()
        out_mat = torch.cat(out_mat).cpu().clone()
        att_weight = att_weight.clone()
    print(f'imgfiles: {inp_mat.shape}, synthesis: {out_mat.shape}, labels: {tar_mat.shape}')
    mat_dict = {'input': inp_mat,
                'target': tar_mat,
                'output': out_mat,
                'att_weight': att_weight,
                'train_weight': train_weight,
                'test_weight': test_weight}
    torch.save(mat_dict, f'mat/{split}/syn_{cfg["dataset"]}.pt')
    print(f'save file: mat/{split}/syn_{cfg["dataset"]}.pt')


def infer(best_cfg, split='PS'):
    print('inference classify stream.......')
    # data grab
    """datasets = get_whole_data(config)
    print(f'load all set {len(datasets[0][0])}')
    best_cfg['n_classes'] = datasets[0][3].size(0) + datasets[0][4].size(0)
    best_cfg['n_train_lbl'] = datasets[0][3].size(0)
    best_cfg['n_test_lbl'] = datasets[0][4].size(0)
    train_linear_weight = F.normalize(datasets[0][3])
    test_linear_weight = F.normalize(datasets[0][4])
    all_set = grab_data(best_cfg, datasets[0][0], datasets[0][1], attr=None, is_train=False, drop_last=False)"""
    if split == 'PS':
        datasets = split_class_ps(best_cfg)
        print(f'load train set {len(datasets[0][0])} test set {len(datasets[0][1])}')
        best_cfg['n_classes'] = datasets[0][5].size(0) + datasets[0][6].size(0)
        best_cfg['n_train_lbl'] = datasets[0][5].size(0)
        best_cfg['n_test_lbl'] = datasets[0][6].size(0)

        train_linear_weight = F.normalize(datasets[0][5])
        test_linear_weight = F.normalize(datasets[0][6])
        test_set = grab_data(best_cfg, datasets[0][1], datasets[0][3], attr=datasets[0][4], is_train=False)
        print('[PS] generate train set and test set.......')
    elif split == 'SS':
        datasets = split_class_ss(best_cfg)
        print(f'load train set {len(datasets[0][0])} test set {len(datasets[0][1])}')
        best_cfg['n_classes'] = datasets[0][4].size(0) + datasets[0][5].size(0)
        best_cfg['n_train_lbl'] = datasets[0][4].size(0)
        best_cfg['n_test_lbl'] = datasets[0][5].size(0)

        train_linear_weight = F.normalize(datasets[0][4])
        test_linear_weight = F.normalize(datasets[0][5])
        test_set = grab_data(best_cfg, datasets[0][1], datasets[0][2], attr=datasets[0][3], is_train=False)
        print('[SS] generate train set and test set.......')
    elif split == 'gen':
        datasets = get_generalize_data(best_cfg)
        print(f'load set {len(datasets[0][0])}')
        best_cfg['n_classes'] = datasets[0][2].size(0)
        best_cfg['n_train_lbl'] = datasets[0][3].size(0)
        best_cfg['n_test_lbl'] = datasets[0][4].size(0)

        train_linear_weight = F.normalize(datasets[0][3])
        test_linear_weight = F.normalize(datasets[0][4])
        att_weight = F.normalize(datasets[0][2])
        test_set = grab_data(best_cfg, datasets[0][0], datasets[0][1], attr=None, is_train=False)
        print('[gen] generate train set and test set......')
    else:
        print('please input valid split (only PS or SS).')
        exit()
    model = models.__dict__['resnet101'](pretrained=True)
    model = ZeroShotCRF(best_cfg, model, Parameter(train_linear_weight), Parameter(test_linear_weight))
    model = nn.DataParallel(model).cuda()  # multi GUP acceleration
    print('create model.......')

    # train model
    best_cfg['train_weight'] = train_linear_weight
    best_cfg['test_weight'] = test_linear_weight

    load_model(best_cfg, model, None, best_cfg['pre_model'])
    if split == 'gen':
        generalize(best_cfg, model, test_set, att_weight, train_linear_weight, test_linear_weight, split=split)
    else:
        generate(best_cfg, model, test_set, train_linear_weight, test_linear_weight, split=split)


if __name__ == '__main__':
    '''for reproducing purpose on CUB:PS ZSL results! '''
    torch.multiprocessing.set_start_method('spawn')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    print('execute main......')
    config = fix_seeds(config_process(parser.parse_args()))
    best_cfg = config
    best_cfg = fix_seeds(best_cfg)

    best_cfg['cycle_len'] = 20
    best_cfg['parts'] = 10
    best_cfg['dropout'] = 0.3
    best_cfg['threshold'] = 0.7
    best_cfg['lr'] = 0.0001
    best_cfg['pre_model'] = './models/Distill-CUB-SS-67.177_checkpoint.pth.tar' 
    infer(best_cfg, split='SS')
