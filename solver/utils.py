#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import numpy as np
import torch
import random
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.parallel
import os


def config_process(config):
    if config.dataset == 'CUB':
        config.image_dir = os.path.join(config.imagedir, 'CUB_200_2011/CUB_200_2011/images/')
        config.image_label = os.path.join('/mnt/samsung/fangzhiyu/VULCAN_Python/NonColliderWeight/data/CUB', 'image_label_PS.txt')
        config.trainval_classes = os.path.join('/mnt/samsung/fangzhiyu/VULCAN_Python/NonColliderWeight/data/CUB', 'trainvalclasses.txt')
        config.test_classes = os.path.join('/mnt/samsung/fangzhiyu/VULCAN_Python/NonColliderWeight/data/CUB', 'testclasses.txt')
        config.all_data = os.path.join('/mnt/samsung/fangzhiyu/tempDataset/CUB_200_2011/CUB_200_2011', 'images.txt')
        # P split
        config.trainval_ps = os.path.join(config.data_root, 'PS', config.txtdir, 'trainval.txt')
        config.train_class_ps = os.path.join(config.data_root, 'PS', config.txtdir, 'trainvalclasses.txt')
        config.test_unseen_ps = os.path.join(config.data_root, 'PS', config.txtdir, 'test_unseen.txt')
        config.test_seen_ps = os.path.join(config.data_root, 'PS', config.txtdir, 'test_seen.txt')
        config.test_class_ps = os.path.join(config.data_root, 'PS', config.txtdir, 'testclasses.txt')
        # standard split
        config.trainval_ss = os.path.join(config.data_root, 'SS', config.txtdir, 'trainval.txt')
        config.train_class_ss = os.path.join(config.data_root, 'SS', config.txtdir, 'trainvalclasses.txt')
        config.test_unseen_ss = os.path.join(config.data_root, 'SS', config.txtdir, 'test_unseen.txt')
        config.test_class_ss = os.path.join(config.data_root, 'SS', config.txtdir, 'testclasses.txt')
        # attribute
        config.attributes_file = os.path.join(config.data_root, 'Attr', config.txtdir, 'class_attributes.txt')
        config.class_file = os.path.join(config.data_root, 'Attr', config.txtdir, 'classes.txt')
        config.attr_dim = 312
        # vit
        config.vit_embed_dim = 312
        config.vit_num_heads = 26
        # mat file
        config.res101_ps = os.path.join('/mnt/samsung/fangzhiyu/tempDataset/GoodBadUgly/PS', 'CUB', 'res101.mat')
        config.att_splits_ps = os.path.join('/mnt/samsung/fangzhiyu/tempDataset/GoodBadUgly/PS', 'CUB', 'att_splits.mat')
        config.res101_ss = os.path.join('/mnt/samsung/fangzhiyu/tempDataset/GoodBadUgly/SS', 'CUB', 'res101.mat')
        config.att_splits_ss = os.path.join('/mnt/samsung/fangzhiyu/tempDataset/GoodBadUgly/SS', 'CUB', 'att_splits.mat')
        config.delete_path = '/BS/Deep_Fragments/work/MSc/CUB_200_2011/CUB_200_2011/images/'

    if config.dataset == 'AWA2':
        config.image_dir = os.path.join(config.imagedir, 'Animals_with_Attributes2/JPEGImages/')
        # P split
        config.trainval_ps = os.path.join(config.data_root, 'PS', config.txtdir, 'trainval.txt')
        config.train_class_ps = os.path.join(config.data_root, 'PS', config.txtdir, 'trainvalclasses.txt')
        config.test_unseen_ps = os.path.join(config.data_root, 'PS', config.txtdir, 'test_unseen.txt')
        config.test_seen_ps = os.path.join(config.data_root, 'PS', config.txtdir, 'test_seen.txt')
        config.test_class_ps = os.path.join(config.data_root, 'PS', config.txtdir, 'testclasses.txt')
        # standard split
        config.trainval_ss = os.path.join(config.data_root, 'SS', config.txtdir, 'trainval.txt')
        config.train_class_ss = os.path.join(config.data_root, 'SS', config.txtdir, 'trainvalclasses.txt')
        config.test_unseen_ss = os.path.join(config.data_root, 'SS', config.txtdir, 'test_unseen.txt')
        config.test_class_ss = os.path.join(config.data_root, 'SS', config.txtdir, 'testclasses.txt')
        # attribute
        config.attributes_file = os.path.join(config.data_root, 'Attr', config.txtdir, 'class_attributes.txt')
        config.class_file = os.path.join(config.data_root, 'Attr', config.txtdir, 'classes.txt')
        config.attr_dim = 85
        # vit
        config.vit_embed_dim = 85
        config.vit_num_heads = 17
        # mat file
        config.res101_ps = os.path.join('/mnt/samsung/fangzhiyu/tempDataset/GoodBadUgly/PS', 'AWA2', 'res101.mat')
        config.att_splits_ps = os.path.join('/mnt/samsung/fangzhiyu/tempDataset/GoodBadUgly/PS', 'AWA2', 'att_splits.mat')
        config.res101_ss = os.path.join('/mnt/samsung/fangzhiyu/tempDataset/GoodBadUgly/SS', 'AWA2', 'res101.mat')
        config.att_splits_ss = os.path.join('/mnt/samsung/fangzhiyu/tempDataset/GoodBadUgly/SS', 'AWA2', 'att_splits.mat')
        config.delete_path = '/BS/xian/work/data/Animals_with_Attributes2//JPEGImages/'

    if config.dataset == 'SUN':
        config.image_dir = os.path.join(config.imagedir, 'SUNAttributeDB/images/')
        config.all_data = os.path.join('/home/fangzhiyu/VULCAN_Python/ZSL_Data/SUNAttributeDB', 'images.txt')
        # P split
        config.trainval_ps = os.path.join(config.data_root, 'PS', config.txtdir, 'trainval.txt')
        config.train_class_ps = os.path.join(config.data_root, 'PS', config.txtdir, 'trainvalclasses.txt')
        config.test_unseen_ps = os.path.join(config.data_root, 'PS', config.txtdir, 'test_unseen.txt')
        config.test_seen_ps = os.path.join(config.data_root, 'PS', config.txtdir, 'test_seen.txt')
        config.test_class_ps = os.path.join(config.data_root, 'PS', config.txtdir, 'testclasses.txt')
        # standard split
        config.trainval_ss = os.path.join(config.data_root, 'SS', config.txtdir, 'trainval.txt')
        config.train_class_ss = os.path.join(config.data_root, 'SS', config.txtdir, 'trainvalclasses.txt')
        config.test_unseen_ss = os.path.join(config.data_root, 'SS', config.txtdir, 'test_unseen.txt')
        config.test_class_ss = os.path.join(config.data_root, 'SS', config.txtdir, 'testclasses.txt')
        # attribute
        config.attributes_file = os.path.join(config.data_root, 'Attr', config.txtdir, 'class_attributes.txt')
        config.class_file = os.path.join(config.data_root, 'Attr', config.txtdir, 'classes.txt')
        config.attr_dim = 102
        # res101 mat
        config.res101_ps = os.path.join(config.data_root, 'PS', config.txtdir, 'res101.mat')
        config.res101_ss = os.path.join(config.data_root, 'SS', config.txtdir, 'res101.mat')
        # vit
        config.vit_embed_dim = 102
        config.vit_num_heads = 17

    if not os.path.exists(config.result_root):
        print('result root not exist')
        os.makedirs(config.result_root)
        print('create result root')
    if not os.path.exists(config.model_root):
        print('model root not exist')
        os.makedirs(config.model_root)
        print('create model root')
    # namespace ==> dictionary
    return vars(config)


def fix_seeds(config):
    seed = config['seed']
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False  # ensure the deterministic
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    cudnn.deterministic = True
    return config


def split_class_ss(config):
    # split attribute
    all_class = {}
    with open(config['class_file'], 'r') as f:
        for line in f.readlines():
            item = line.strip().split()
            all_class[item[-1]] = int(item[0])-1
    with open(config['train_class_ss'], 'r') as f:
        train_class = [l.strip() for l in f.readlines()]
    train_class_idx = [all_class[x] for x in train_class]
    with open(config['test_class_ss'], 'r') as f:
        test_class = [l.strip() for l in f.readlines()]
    test_class_idx = [all_class[x] for x in test_class]
    # print(f'{train_class_idx}\n{test_class_idx}\n{all_class}')
    attributes = np.loadtxt(config['attributes_file'], dtype='float')
    train_attr = torch.from_numpy(attributes[train_class_idx, :]).float()
    test_attr = torch.from_numpy(attributes[test_class_idx, :]).float()
    attributes = np.around(attributes / 10).astype(int)

    label_map = {}
    attr_map = {}
    trainval = []
    with open(config['trainval_ss'], 'r') as f:
        for line in f.readlines():
            item = line.strip().split()
            if config['dataset'] == 'SUN':
                name_list = item[0].split("/")
                if len(name_list) > 3:
                    classname = f'{name_list[1]}_{name_list[2]}'
                    cid = train_class.index(classname)
                else:
                    cid = train_class.index(item[0].split("/")[1])
            else:
                cid = train_class.index(item[0].split("/")[0])
            label_map[item[0]] = cid
            attr_map[item[0]] = attributes[cid]
            trainval.append(item[0])
    test = []
    with open(config['test_unseen_ss'], 'r') as f:
        for line in f.readlines():
            item = line.strip().split()
            if config['dataset'] == 'SUN':
                name_list = item[0].split("/")
                if len(name_list) > 3:
                    classname = f'{name_list[1]}_{name_list[2]}'
                    cid = test_class.index(classname)
                else:
                    cid = test_class.index(item[0].split("/")[1])
            else:
                cid = test_class.index(item[0].split("/")[0])
            label_map[item[0]] = cid
            attr_map[item[0]] = attributes[cid]
            test.append(item[0])
    # print(f'{len(trainval)}  {len(test)}  {len(label_map)}  {train_attr.shape}  {test_attr.shape}')
    return [(trainval, test, label_map, attr_map, train_attr, test_attr)]


def split_class_ps(config):
    # split attribute
    all_class = {}
    with open(config['class_file'], 'r') as f:
        for line in f.readlines():
            item = line.strip().split()
            all_class[item[-1]] = int(item[0])-1
    with open(config['train_class_ps'], 'r') as f:
        train_class = [l.strip() for l in f.readlines()]
    train_class_idx = [all_class[x] for x in train_class]
    with open(config['test_class_ps'], 'r') as f:
        test_class = [l.strip() for l in f.readlines()]
    test_class_idx = [all_class[x] for x in test_class]
    # print(f'{train_class_idx}  {len(train_class)}\n{test_class_idx}  {len(test_class)}\n{all_class}')
    attributes = np.loadtxt(config['attributes_file'], dtype='float')
    # attributes = (attributes-attributes.mean())/attributes.std()
    train_attr = torch.from_numpy(attributes[train_class_idx, :]).float()
    test_attr = torch.from_numpy(attributes[test_class_idx, :]).float()
    # attributes = np.around(attributes/10).astype(int)  # 将属性变成整数标签

    label_map = {}
    attr_map = {}
    trainval = []
    with open(config['trainval_ps'], 'r') as f:
        for line in f.readlines():
            item = line.strip().split()
            if config['dataset'] == 'SUN':
                name_list = item[0].split("/")
                if len(name_list) > 3:
                    classname = f'{name_list[1]}_{name_list[2]}'
                    cid = train_class.index(classname)
                else:
                    cid = train_class.index(item[0].split("/")[1])
            else:
                cid = train_class.index(item[0].split("/")[0])
            label_map[item[0]] = cid
            attr_map[item[0]] = attributes[cid]
            trainval.append(item[0])
    test_unseen = []
    with open(config['test_unseen_ps'], 'r') as f:
        for line in f.readlines():
            item = line.strip().split()
            if config['dataset'] == 'SUN':
                name_list = item[0].split("/")
                if len(name_list) > 3:
                    classname = f'{name_list[1]}_{name_list[2]}'
                    cid = test_class.index(classname)
                else:
                    cid = test_class.index(item[0].split("/")[1])
            else:
                cid = test_class.index(item[0].split("/")[0])
            label_map[item[0]] = cid
            attr_map[item[0]] = attributes[cid]
            test_unseen.append(item[0])
    test_seen = []
    with open(config['test_seen_ps'], 'r') as f:
        for line in f.readlines():
            item = line.strip().split()
            if config['dataset'] == 'SUN':
                name_list = item[0].split("/")
                if len(name_list) > 3:
                    classname = f'{name_list[1]}_{name_list[2]}'
                    cid = train_class.index(classname)
                else:
                    cid = train_class.index(item[0].split("/")[1])
            else:
                cid = train_class.index(item[0].split("/")[0])
            label_map[item[0]] = cid
            attr_map[item[0]] = attributes[cid]
            test_seen.append(item[0])
    # print(f'{len(trainval)}  {len(test_unseen) + len(test_seen)}  {train_attr.shape}  {test_attr.shape}')
    # print(f'{set(train_label)}\n{set(test_label)}')
    # exit()
    return [(trainval, test_unseen, test_seen, label_map, attr_map, train_attr, test_attr)]


def get_whole_data(config):
    # split attribute
    all_class = {}
    with open(config['class_file'], 'r') as f:
        for line in f.readlines():
            item = line.strip().split()
            all_class[item[-1]] = int(item[0])-1
    with open(config['train_class_ps'], 'r') as f:
        train_class = [l.strip() for l in f.readlines()]
    train_class_idx = [all_class[x] for x in train_class]
    with open(config['test_class_ps'], 'r') as f:
        test_class = [l.strip() for l in f.readlines()]
    test_class_idx = [all_class[x] for x in test_class]
    # print(f'{train_class_idx}  {len(train_class)}\n{test_class_idx}  {len(test_class)}\n{all_class}')
    attributes = np.loadtxt(config['attributes_file'], dtype='float')
    # attributes = (attributes-attributes.mean())/attributes.std()
    train_attr = torch.from_numpy(attributes[train_class_idx, :]).float()
    test_attr = torch.from_numpy(attributes[test_class_idx, :]).float()
    # attributes = np.around(attributes/10).astype(int)  # 将属性变成整数标签

    label_map = {}
    attr_map = {}
    alldata = []
    with open(config['all_data'], 'r') as f:
        for line in f.readlines():
            item = line.strip().split()
            cid = all_class[item[-1].split("/")[0]]
            label_map[item[-1]] = cid
            attr_map[item[-1]] = attributes[cid]
            alldata.append(item[-1])
    print(f'{np.size(alldata)}  {np.size(label_map)}  {train_attr.shape}  {test_attr.shape}')
    exit()
    return [(alldata, label_map, attr_map, train_attr, test_attr)]


def get_generalize_data(config):
    # split attribute
    all_class = {}
    with open(config['class_file'], 'r') as f:
        for line in f.readlines():
            item = line.strip().split()
            all_class[item[-1]] = int(item[0])-1
    with open(config['train_class_ps'], 'r') as f:
        train_class = [l.strip() for l in f.readlines()]
    train_class_idx = [all_class[x] for x in train_class]
    with open(config['test_class_ps'], 'r') as f:
        test_class = [l.strip() for l in f.readlines()]
    test_class_idx = [all_class[x] for x in test_class]
    # print(f'{train_class_idx}  {len(train_class)}\n{test_class_idx}  {len(test_class)}\n{all_class}')
    attributes = np.loadtxt(config['attributes_file'], dtype='float')
    # attributes = (attributes-attributes.mean())/attributes.std()
    train_attr = torch.from_numpy(attributes[train_class_idx, :]).float()
    test_attr = torch.from_numpy(attributes[test_class_idx, :]).float()
    # attributes = np.around(attributes/10).astype(int)  # 将属性变成整数标签
    attr_map = torch.from_numpy(attributes).float()

    label_map = {}
    test_unseen = []
    with open(config['test_unseen_ps'], 'r') as f:
        for line in f.readlines():
            item = line.strip().split()
            if config['dataset'] == 'SUN':
                name_list = item[0].split("/")
                if len(name_list) > 3:
                    classname = f'{name_list[1]}_{name_list[2]}'
                    cid = all_class[classname]
                else:
                    cid = all_class[item[0].split("/")[1]]
            else:
                cid = all_class[item[0].split("/")[0]]
            label_map[item[0]] = cid
            test_unseen.append(item[0])
    test_seen = []
    with open(config['test_seen_ps'], 'r') as f:
        for line in f.readlines():
            item = line.strip().split()
            if config['dataset'] == 'SUN':
                name_list = item[0].split("/")
                if len(name_list) > 3:
                    classname = f'{name_list[1]}_{name_list[2]}'
                    cid = all_class[classname]
                else:
                    cid = all_class[item[0].split("/")[1]]
            else:
                cid = all_class[item[0].split("/")[0]]
            label_map[item[0]] = cid
            test_seen.append(item[0])
    # print(f'{np.shape(test_unseen), np.shape(test_seen)}')
    alldata = np.concatenate((test_unseen, test_seen))
    # print(f'{np.shape(alldata)}')
    # exit()
    # print(f'{len(trainval)}  {len(test_unseen) + len(test_seen)}  {train_attr.shape}  {test_attr.shape}')
    # print(f'{set(train_label)}\n{set(test_label)}')
    return [(alldata, label_map, attr_map, train_attr, test_attr)]


def dict_str(config, tf=False):
    return 'author did not implement this function.'
