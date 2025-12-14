import os
import re
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

import biformer
import FCN

from models.block.Base import ChannelChecker
import FCNHead
from collections import OrderedDict
from typing import Dict
from common import ScaleInOutput
from torchvision.models.feature_extraction import create_feature_extractor
from mmseg.models import build_head

def BasicConv(filter_in, filter_out, kernel_size, stride=1, pad=None):
    if not pad:
        pad = (kernel_size - 1) // 2 if kernel_size else 0
    else:
        pad = pad
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.ReLU(inplace=True)),
    ]))

class ChangeDetection(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.inplanes = int(re.sub(r"\D", "", opt.backbone.split("_")[-1])) 
        # print(self.inplanes )
     
        self.ame=biformer_base()
        norm_cfg = dict(type='LN', requires_grad=True)
        decode_head = dict(
            type='MusterHead',
            embed_dims=1024,
            patch_size=4,
            window_size=12,
            mlp_ratio=4,
            depths=(2, 2, 2, 2),
            num_heads=(32, 16, 8, 4),
            strides=(2, 2, 2, 4),

            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            act_cfg=dict(type='GELU'),
            # norm_cfg=dict(type='LN'),
            with_cp=False,
            init_cfg=None,
            in_channels=[1024, 512, 256, 128],
            in_index=[0, 1, 2, 3],
            # pool_scales=(1, 2, 3, 6),
            channels=256,
            # dropout_ratio=0.1,
            num_classes=self.inplanes,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
        self.lgd = build_head(decode_head)
        self.head=FCNHead(self.inplanes, 2)
        self.stage1_Conv1 = BasicConv(self.inplanes * 2, self.inplanes, 1)
        self.stage2_Conv1 = BasicConv(self.inplanes * 4, self.inplanes * 2, 1)
        self.stage3_Conv1 = BasicConv(self.inplanes * 8, self.inplanes * 4, 1)
        self.stage4_Conv1 = BasicConv(self.inplanes * 16, self.inplanes * 8, 1)

    def forward(self, xa, xb, tta=False):
        
        _, _, h_input, w_input = xa.shape
        assert xa.shape == xb.shape, "The two images are not the same size, please check it."

        fa1, fa2, fa3, fa4 = self.ame(xa)  

        fb1, fb2, fb3, fb4 = self.ame(xb)


        f1 = self.stage1_Conv1(torch.cat([fa1, fb1], 1))  # inplanes
        f2 = self.stage2_Conv1(torch.cat([fa2, fb2], 1))  # inplanes * 2
        f3 = self.stage3_Conv1(torch.cat([fa3, fb3], 1))  # inplanes * 4
        f4 = self.stage4_Conv1(torch.cat([fa4, fb4], 1))  # inplanes * 8
        ms_feats = f1, f2, f3, f4
        change = self.lgd(ms_feats)
        out = self.head_forward(change, out_size=(h_input, w_input))
        return out
    def _init_weight(self, pretrain=''): 
            for m in self.modules():
                if isinstance(m, nn.Conv2d): 
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):  
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            if pretrain.endswith('.pt'):
                pretrained_dict = torch.load(pretrain)
                if isinstance(pretrained_dict, nn.DataParallel):
                    pretrained_dict = pretrained_dict.module
                model_dict = self.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.state_dict().items()
                                if k in model_dict.keys()}
                model_dict.update(pretrained_dict)
                self.load_state_dict(OrderedDict(model_dict), strict=True)




class EnsembleSTLNet(nn.Module):
    def __init__(self, ckp_paths, device, method="avg2", input_size=448):
        super(EnsembleHATNet, self).__init__()
        self.method = method
        self.models_list = []
        for ckp_path in ckp_paths:
            if os.path.isdir(ckp_path):
                weight_file = os.listdir(ckp_path)
                ckp_path = os.path.join(ckp_path, weight_file[0])
            print("--Load model: {}".format(ckp_path))
            model = torch.load(ckp_path, map_location=device)
            if isinstance(model, torch.nn.parallel.DistributedDataParallel) \
                    or isinstance(model, nn.DataParallel):
                model = model.module
            self.models_list.append(model)
        self.scale = ScaleInOutput(input_size)

    def eval(self):
        for model in self.models_list:
            model.eval()

    def forward(self, xa, xb):
        xa, xb = self.scale.scale_input((xa, xb))
        out1 = 0
        cd_pred = None

        for i, model in enumerate(self.models_list):
            outs = model(xa, xb)
            if not isinstance(outs, tuple):
                outs = (outs, outs)
            outs = self.scale.scale_output(outs)
            if "avg" in self.method:
                if self.method == "avg2":
                    outs = (F.softmax(outs[0], dim=1), F.softmax(outs[1], dim=1))
                out1 += outs[0]
                _, cd_pred = torch.max(out1, 1)

        return cd_pred