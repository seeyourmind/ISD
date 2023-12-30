#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
from backbone.classifier import Classifier


class CALayer(nn.Module):
    # Channel Attention Layer
    def __init__(self, channel):
        super(CALayer, self).__init__()
        # feature channel downscale and upscale --> channel weight
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Softmax(dim=1),
            nn.Conv2d(channel, channel // 2, 1),
            nn.ReLU(),
            nn.Conv2d(channel // 2, channel, 1),
            nn.Sigmoid())

    def forward(self, x):
        return x * self.attention(x)


class RCAB(nn.Module):
    # Residual Channel Attention Block
    def __init__(self, in_channel, out_channel, res_scale=1, ispool=True):
        super(RCAB, self).__init__()
        self.res = nn.Sequential(
            nn.Conv2d(in_channel, in_channel//2, 1),
            nn.ReLU(),
            nn.Conv2d(in_channel//2, in_channel//2, 3),
            nn.ReLU(),
            nn.Conv2d(in_channel//2, out_channel, 1),
            CALayer(out_channel)
        )
        self.pool = nn.MaxPool2d(5, 5)
        self.ispool = ispool
        self.res_scale = res_scale

    def forward(self, x):
        res = self.res(x).mul(self.res_scale)
        b, d, _, _ = res.size()
        res = self.pool(res).squeeze()

        return res


class ZeroShotCRF(nn.Module):
    def __init__(self, cfg, model, train_attr, test_attr):
        super(ZeroShotCRF, self).__init__()
        self.cfg = cfg
        self.map_size = 7
        self.cov_channel = 2048
        self.bs = self.cfg['batch_size']
        self.map_threshold = self.cfg['threshold']
        self.parts = self.cfg['parts']
        drop_out = self.cfg['dropout']
        d = self.cfg['attr_dim']
        c = self.cov_channel

        self.pre_features = nn.Sequential(*list(model.children())[:-2])
        self.e_classifier = Classifier(cfg, train_attr, test_attr)
        # self.t_classifier = Classifier(cfg, train_attr, test_attr, False)

        """Emission Structure"""
        # AREN attention
        self.pool = nn.MaxPool2d(self.map_size, self.map_size)
        self.cov = nn.Sequential(nn.Conv2d(self.cov_channel, self.cov_channel//2, 1),
                                 nn.ReLU(),
                                 nn.Conv2d(self.cov_channel//2, self.parts, 1))
        self.p_linear = nn.Linear(self.cov_channel * self.parts, d, False)
        if drop_out is None:
            self.dropout = nn.Identity()
        else:
            self.dropout = nn.Dropout(drop_out)

        """Transition Structure"""
        # Transition
        self.conv_compress = nn.Conv2d(c, d, 1)
        self.pool_compress = nn.MaxPool2d(self.map_size, self.map_size)
        self.ca_block = RCAB(d, d)

        # fusion factor
        self.alpha = nn.Parameter(torch.tensor(1.0)).cuda()
        self.fusion = nn.Linear(d*d, d)

    def emission(self, feats):
        """Forward"""
        w = feats.size()
        weights = torch.sigmoid(self.cov(feats))
        # batch, parts, width, height = weights.size()
        # weights_layout = weights.view(batch, -1)
        # threshold_value, _ = weights_layout.max(dim=1)  # AT_max => the global maximum for all masks
        # local_max, _ = weights.view(batch, parts, -1).max(dim=2)  # m_v(k) => local max for each mask
        # threshold_value = self.map_threshold * threshold_value.view(batch, 1).expand(batch, parts)  # T = alpha*AT_max
        # weights = weights * local_max.ge(threshold_value).view(batch, parts, 1, 1).float().expand(batch, parts, width, height)  # purified mask

        blocks = []  # v_ARE without concat
        for k in range(self.parts):
            Y = feats * weights[:, k, :, :]. unsqueeze(dim=1).expand(w[0], self.cov_channel, w[2], w[3])  # T(R(Z))
            blocks.append(self.pool(Y).squeeze().view(-1, self.cov_channel))

        output = self.dropout(self.p_linear(torch.cat(blocks, dim=1)))
        return output

    def transition_ca(self, feats):
        tw = self.conv_compress(feats)
        # prob = self.pool_compress(tw).squeeze()
        tw = self.ca_block(tw)
        out = tw
        return out

    def get_fusion(self, x):
        features = self.pre_features(x)
        emission_score = self.emission(features)
        transition_score = self.transition_ca(features)
        out = torch.matmul(emission_score.unsqueeze(2), transition_score.unsqueeze(1)).view(self.bs, -1)
        out = self.fusion(out)
        return out

    def forward(self, x, is_seen=True):
        out = self.get_fusion(x)
        # eout = self.e_classifier(emission_score)
        # tout = self.e_classifier(transition_score)
        eout = self.e_classifier(out, is_seen)

        return eout, out#eout#eout tout
