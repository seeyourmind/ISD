# #!/usr/bin/env python
# # -*- encoding: utf-8 -*-
import torchvision.models as models
import argparse
zsl_data = '/mnt/samsung/fangzhiyu/tempDataset'
root = '/mnt/samsung/fangzhiyu/VULCAN_Python/ZSCRF-distill'
# root = '/home/fangzhiyu/VULCAN_Python/ZSCRF-distill'
# zsl_data = '/home/fangzhiyu/VULCAN_Python/ZSL_Data'
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description='source example code for causation-A')
# setting source files and directory
parser.add_argument('--dataset', default='CUB', choices=['CUB', 'AWA2', 'SUN'], metavar='NAME', help='dataset name')
parser.add_argument('--imagedir', default=zsl_data, metavar='NAME', help='image dir for loading images')
parser.add_argument('--txtdir', default='CUB', choices=['CUB', 'AWA2', 'SUN'], metavar='NAME', help='dirs with texfiles included')
parser.add_argument('--data_root', default=root+'/data', type=str, metavar='DIRECTORY', help='path to data directory')
parser.add_argument('--model_root', default=root+'/models', metavar='DIRECTORY', help='dataset to model directory')
parser.add_argument('--result_root', default=root+'/results', metavar='DIRECTORY', help='dataset to result directory')
# parser.add_argument('--pre_model', default=root+'/models/CUB-PS-P_model_best.pth.tar', type=str, metavar='FILE', help='the path for pretrained model')
parser.add_argument('--pre_model', default=root+'/source/PS/CUB-PS-P_69.346_checkpoint.pth.tar', type=str, metavar='FILE', help='the path for pretrained model')
# # ViT model setting
# parser.add_argument('--vit_img_size', default=224, type=int, metavar='NUM', help='ViT initialization: img_size')
# parser.add_argument('--vit_embed_dim', default=768, type=int, metavar='NUM', help='ViT initialization: embed_dim')
# parser.add_argument('--vit_num_heads', default=12, type=int, metavar='NUM', help='ViT initialization: num_heads')
# parser.add_argument('--vit_no_head', default=True, type=bool, metavar='FLAG', help='ViT initialization: no_head')
# # CRF model setting
# parser.add_argument('--crf_nb_labels', default=11, type=int, metavar='NUM', help='CRF initialization: nb_labels')
# parser.add_argument('--crf_bos_tag', default=0, type=int, metavar='NUM', help='CRF initialization: bos_tag')
# parser.add_argument('--crf_eos_tag', default=10, type=int, metavar='NUM', help='CRF initialization: eos_tag')
# # setting global training parameters
parser.add_argument('--batch_size', default=64, type=int, metavar='NUM', help='batch size')
# parser.add_argument('--dropout', default=0.3, type=float, metavar='NUM', help='dropout')
# parser.add_argument('--fix_feature', default=0, type=int, metavar='NUM', help='freeze base feature layer weight')
# # setting Classify-stream parameters
# parser.add_argument('--folds', default=5, type=int, metavar='NUM', help='folders')
# parser.add_argument('--cpp_map', default=20, type=int, metavar='NUM', help='cpp_map is the number of compressed feature maps')
# parser.add_argument('--sample_num', default=10, type=int, metavar='NUM', help='the number of sample A')

# parser.add_argument('--aparts', default=20,  type=int, metavar='NUM', help='attention parts')
# parser.add_argument('--tparts', default=20,  type=int, metavar='NUM', help='attention parts')
# # setting Classify-stream train parameters
parser.add_argument('--epochs', default=200, type=int, metavar='NUM', help='epochs is the total epochs')
parser.add_argument('--start_epoch', default=0, type=int, metavar='NUM', help='#start epoch number')
parser.add_argument('--final_epoch', default=200, type=int, metavar='NUM', help='last number of epoch')
parser.add_argument('--seed', default=131, type=int, metavar='NUM', help='random seeds')
# # parser.add_argument('--step', default=20, type=int, metavar='NUM', help='for SGD the default lr reducing step')
# parser.add_argument('--pretrain', default=1, type=int, choices=[0, 1], metavar='FLAG', help='0: [train]use imagenet model, 1: [test]use pretarined model')
# parser.add_argument('--fix_amu', default=0, choices=[0, 1], type=int, metavar='FLAG', help='fix the lr or not')

# # parser.add_argument('--cycle_mul', default=2,  type=int, metavar='NUM', help='cycle_mul')
parser.add_argument('--print_freq', default=10, type=int, metavar='NUM', help='print frequency')
# parser.add_argument('--determine', default=1, choices=[0, 1], type=int, metavar='FLAG', help='for reproduce the results')
parser.add_argument('--output', default='CUB-PS-IZSwSD', type=str, metavar='DIRECTORY', help='name of output')
parser.add_argument('--is_train', default=1, choices=[0, 1], type=int, metavar='FLAG', help='choose train attr-mat or test attr-mat')
# parser.add_argument('--is_valid', default=False, type=bool, metavar='NUM', help='PS is True, SS is False')
parser.add_argument('--ls_coef_part', default=1, type=int, metavar='NUM', help='part coef for loss')
parser.add_argument('--ls_coef_bi', default=1, type=int, metavar='NUM', help='bilinear coef for loss')
parser.add_argument('--coef_part', default=1, type=int, metavar='NUM', help='part coef for prediction')
parser.add_argument('--coef_bi', default=1, type=int, metavar='NUM', help='bilinear coef for prediction')

# adjust super-params == optimizer
parser.add_argument('--lr', default='0.0001',  type=float, metavar='RANGE', help='lr learning rate')
parser.add_argument('--momentum', default=0.9,  type=float, metavar='NUM', help='momentum')
parser.add_argument('--weight_decay', default=0.0005,  type=float, metavar='NUM', help='weight decay')
parser.add_argument('--lr_strategy', default='sgdr_lr', type=str, choices=['sgdr_lr', 'step_lr'], metavar='METHOD', help='adjust lr type')
parser.add_argument('--cycle_len', default=10,  type=int, metavar='NUM', help='cycle_len')
# adjust super-params == model
parser.add_argument('--parts', default=20,  type=int, metavar='NUM', help='attention parts')
# parser.add_argument('--threshold', default=0.8, type=float, metavar='NUM', help='attention filter threshold')
