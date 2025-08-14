import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import copy
import time
import argparse
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights
from torch.utils.data import DataLoader
from utils.func import AvgMeter, clip_gradient
from utils.lr_scheduler import get_scheduler
from utils.dataset import MyDataset
from utils.metric_tool import ConfuseMatrixMeter
from utils.loss_f import BCEDICE_loss

from models.SMOW_Net import SMOW_Net
from models.SMOW_Net_LW import SMOW_Net_LW

from compare.FC_EF import FC_EF
from compare.SNUNet import SNUNet
from compare.DTCDSCN import DTCDSCN
from compare.ChangeFormerV6 import ChangeFormerV6
from compare.A2Net import A2Net
from compare.IFN import DSIFN
from compare.TFI_GR import TFI_GR
from compare.BIT import BIT
from compare.PA_Former import PA_Former
from compare.AFCF3D_NET import AFCF3D_NET
from compare.SEIFNet import SEIFNet
from compare.ELGCNet import ELGCNet
from compare.rs_mamba import RSM_CD
from compare.change_mamba import Changemamba
from compare.cd_mamba import CDMamba


# set seeds
def seed_torch(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
seed_torch(2022)


def parse_option():
    parser = argparse.ArgumentParser()
    # data set
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--trainsize', type=int, default=256)
    parser.add_argument('--data_dir', type=str, default='/raid/SMOW-Net/datasets/GVLM-CD-256')
    # training
    parser.add_argument('--epochs', type=int, default=200, help='epoch number')
    parser.add_argument('--optim', type=str, default='adamW', help='optimizer')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['step', 'cosine'])
    parser.add_argument('--warmup_epoch', type=int, default=-1, help='warmup epoch')
    parser.add_argument('--warmup_multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--lr_decay_epochs', type=int, default=[120, 160, 200], nargs='+', help='for step scheduler')
    parser.add_argument('--lr_decay_steps', type=int, default=20, help='for step scheduler.step size to decay lr')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='for step scheduler.decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    # io
    parser.add_argument('--output_dir', type=str, default='/raid/SMOW-Net/checkpoints', help='output director')
    opt, unparsed = parser.parse_known_args()

    return opt

def build_loader(opt):
    train_data = MyDataset(opt.data_dir, "train")
    train_loader = DataLoader(train_data, batch_size=opt.batchsize, shuffle=True, num_workers=8, pin_memory=True)
    val_data = MyDataset(opt.data_dir, "val")
    val_loader = DataLoader(val_data, batch_size=opt.batchsize, shuffle=False, num_workers=8, pin_memory=True)
    return train_loader, val_loader

def build_model(opt):
    resnet18 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model = SMOW_Net(copy.deepcopy(resnet18))
    # SMOW_Net: 
    # resnet18 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    # model = SMOW_Net(copy.deepcopy(resnet18))
    # SMOW_Net_LW: 
    # model = SMOW_Net_LW()
    # FC_EF: 
    # model = FC_EF(input_nbr=3, label_nbr=2)
    # BIT: 
    # model = BIT(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4, with_pos='learned', enc_depth=1, dec_depth=8, decoder_dim_head=8)
    # PA_Former: 
    # model = PA_Former(n_class=2)
    # AFCF3D_NET: 
    # resnet = torchvision.models.resnet18(pretrained=True)
    # model = AFCF3D_NET(32, copy.deepcopy(resnet))
    # SEIFNet：
    # model = SEIFNet(input_nc=3, output_nc=2)
    # TFI_GR：
    # model = TFI_GR(3, 1)
    # ELGCNet：
    # model = ELGCNet(dec_embed_dim=256)
    # ChangeFormerV6：
    # model = ChangeFormerV6(embed_dim=256)
    # SNUNet：
    # model = SNUNet(in_ch=3, out_ch=2)
    # DTCDSCN：
    # model = DTCDSCN(in_channels=3)
    # A2Net：
    # model = A2Net(3, 1)
    # IFN：
    # model = DSIFN()
    # RS_Mamba
    # model = RSM_CD(drop_path_rate=0.2, dims=96, depths=[ 2, 2, 9, 2 ], ssm_d_state=16, ssm_dt_rank="auto", ssm_ratio=2.0, mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", image_size=256, downsample_raito=1)
    # Change_Mamba
    # model = Changemamba(device=torch.device("cuda:0"), pretrained="", patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 9, 2], dims=96, ssm_d_state=16, ssm_ratio=2.0, ssm_rank_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu", ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2", mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, drop_path_rate=0.1, patch_norm=True, norm_layer='ln', downsample_version="v2", patchembed_version="v2", gmlp=False, use_checkpoint=False)
    # CD_Mamba
    # model = CDMamba(spatial_dims=2, in_channels=3, init_filters=16, out_channels=2,mode="AGLGF", conv_mode="orignal_dinner", up_mode="SRCM", up_conv_mode="deepwise", norm=["GROUP", {"num_groups": 8}],blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1], resdiual=False,diff_abs="later", stage=2,mamba_act="relu", local_query_model="orignal_dinner")
    model = model.cuda()
    return model

def main(opt):
    train_loader, val_loader = build_loader(opt)
    n_data = len(train_loader.dataset)
    print(f"length of training dataset: {n_data}\n")
    print(f"length of val dataset: {len(val_loader.dataset)}\n")
    model = build_model(opt)

    CE = BCEDICE_loss

    if opt.optim == 'adamW':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr, weight_decay=opt.weight_decay)
    else:
        raise NotImplementedError
    scheduler = get_scheduler(optimizer, len(train_loader), opt)
    # routine
    iou = 0.0
    for epoch in range(1, opt.epochs + 1):
        tic = time.time()
        tool4metric = ConfuseMatrixMeter(n_class=2)
        train(train_loader, model, optimizer, CE, scheduler, epoch, tool4metric)
        print('epoch {}, total time {:.2f}, learning_rate {}'.format(epoch, (time.time() - tic), optimizer.param_groups[0]['lr']))
        print('begin val')
        val(val_loader, model, CE, epoch, tool4metric)
        print('epoch {}, total time {:.2f}'.format(epoch, (time.time() - tic)))
        scores_dictionary = tool4metric.get_scores()
        best_iou = scores_dictionary['iou']
        if best_iou >= iou:
            iou = best_iou
            torch.save(model.state_dict(), os.path.join(opt.output_dir, f"best.pth"))
            print("model saved {}!".format(os.path.join(opt.output_dir, f"best.pth")))


def train(train_loader, model, optimizer, criterion, scheduler, epoch, tool4metric):
    tool4metric.clear()
    model.train()
    loss_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        imageA, imageB, gts = pack
        imageA = imageA.cuda().float()
        imageB = imageB.cuda().float()
        gts = gts.cuda().float()

        # forward
        pred_s = model(imageA, imageB)
        if pred_s.size(1) == 1:
            pred_s = pred_s.squeeze(1)
        else:
            pred_s = torch.sigmoid(pred_s)
            pred_s = pred_s[:, 1, :, :]
        loss = criterion(pred_s, gts)
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        scheduler.step()

        loss_record.update(loss.data, opt.batchsize)
        bin_preds_mask = (pred_s.to('cpu') > 0.5).detach().numpy().astype(int)
        mask = gts.to('cpu').numpy().astype(int)

        tool4metric.update_cm(pr=bin_preds_mask, gt=mask)

        if i % 100 == 0 or i == len(train_loader):
            print('Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}],'
                  'Loss: {:.4f}'.format(epoch, opt.epochs, i, len(train_loader), loss_record.show()))

    scores_dictionary = tool4metric.get_scores()
    print("IoU for epoch {} is {}".format(epoch, scores_dictionary["iou"]))
    print("F1 for epoch {} is {}".format(epoch, scores_dictionary["F1"]))
    print("acc for epoch {} is {}".format(epoch, scores_dictionary["acc"]))
    print("precision for epoch {} is {}".format(epoch, scores_dictionary["precision"]))
    print("recall for epoch {} is {}".format(epoch, scores_dictionary["recall"]))
    print('---------------------------------------------')
    filepath1 = os.path.join(opt.output_dir, 'train.txt')
    with open(filepath1, 'a') as file:
        file.write(f"Epoch: {epoch}, IoU: {scores_dictionary['iou']:.4f}\n")
        file.write(f"Epoch: {epoch}, F1: {scores_dictionary['F1']:.4f}\n")
        file.write(f"Epoch: {epoch}, acc: {scores_dictionary['acc']:.4f}\n")
        file.write(f"Epoch: {epoch}, precision: {scores_dictionary['precision']:.4f}\n")
        file.write(f"Epoch: {epoch}, recall: {scores_dictionary['recall']:.4f}\n")

def val(val_loader, model, criterion, epoch, tool4metric):
    model.eval()
    tool4metric.clear()
    loss_record = AvgMeter()
    with torch.no_grad():
        for i, pack in enumerate(val_loader):
            imageA, imageB, gts = pack
            imageA = imageA.cuda().float()
            imageB = imageB.cuda().float()
            gts = gts.cuda().float()

            pred_s = model(imageA, imageB)
            if pred_s.size(1) == 1:
                pred_s = pred_s.squeeze(1)
            else:
                pred_s = torch.sigmoid(pred_s)
                pred_s = pred_s[:, 1, :, :]
            loss = criterion(pred_s, gts)

            bin_preds_mask = (pred_s.to('cpu') > 0.5).detach().numpy().astype(int)
            mask = gts.to('cpu').numpy().astype(int)
            tool4metric.update_cm(pr=bin_preds_mask, gt=mask)
            loss_record.update(loss.data, opt.batchsize)

            if i % 100 == 0 or i == len(val_loader):
                print('Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}],'
                      'Loss: {:.4f}'.format(epoch, opt.epochs, i, len(val_loader), loss_record.show()))

        scores_dictionary = tool4metric.get_scores()
        print("IoU for epoch {} is {}".format(epoch, scores_dictionary["iou"]))
        print("F1 for epoch {} is {}".format(epoch, scores_dictionary["F1"]))
        print("acc for epoch {} is {}".format(epoch, scores_dictionary["acc"]))
        print("precision for epoch {} is {}".format(epoch, scores_dictionary["precision"]))
        print("recall for epoch {} is {}".format(epoch, scores_dictionary["recall"]))
        print('---------------------------------------------')
        filepath2 = os.path.join(opt.output_dir, 'val.txt')
        with open(filepath2, 'a') as file:
            file.write(f"Epoch: {epoch}, IoU: {scores_dictionary['iou']:.4f}\n")
            file.write(f"Epoch: {epoch}, F1: {scores_dictionary['F1']:.4f}\n")
            file.write(f"Epoch: {epoch}, acc: {scores_dictionary['acc']:.4f}\n")
            file.write(f"Epoch: {epoch}, precision: {scores_dictionary['precision']:.4f}\n")
            file.write(f"Epoch: {epoch}, recall: {scores_dictionary['recall']:.4f}\n")

if __name__ == '__main__':
    opt = parse_option()
    main(opt)