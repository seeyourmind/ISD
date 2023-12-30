#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, cfg, train_attr, test_attr, dropout=True):
        super(Classifier, self).__init__()

        # weight => A
        c1, d = train_attr.size()
        c2, _ = test_attr.size()
        self.train_linear = nn.Linear(d, c1, False)
        self.train_linear.weight = train_attr
        for para in self.train_linear.parameters():
            para.requires_grad = False

        self.test_linear = nn.Linear(d, c2, False)
        self.test_linear.weight = test_attr
        for para in self.test_linear.parameters():
            para.requires_grad = False

        self.pre_layer = nn.ModuleList([self.train_linear, self.test_linear])

        # Dropout
        if cfg['dropout'] is None or not dropout:
            self.dropout = nn.Identity()
        else:
            self.dropout = nn.Dropout(cfg['dropout'])

    def forward(self, feats, is_seen=True):

        if is_seen:
            out = self.pre_layer[0](self.dropout(feats))
        else:
            out = self.pre_layer[1](self.dropout(feats))

        return out

