import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import argparse
import copy
import cv2
import torch
import tqdm
import numpy as np
import torchvision
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights
from torch.utils.data import DataLoader
from utils.metric_tool import ConfuseMatrixMeter
from utils.dataset import MyDataset
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/raid/SMOW-Net/checkpoints/best.pth', help='path to model file')
    parser.add_argument('--testsize', type=int, default=256, help='testing size')
    parser.add_argument('--test_datasets', type=str, default=['NJU2000-test'], nargs='+', help='test dataset')
    parser.add_argument('--data_path', type=str, default='/raid/SMOW-Net/datasets/GVLM-CD-256')
    parser.add_argument('--save_path', type=str, help='test dataset')
    parser.add_argument('--multi_load', action='store_true', help='whether to load multi-gpu weight')
    opt = parser.parse_args()


    test_data = MyDataset(opt.data_path, "test")
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=8)

    if opt.save_path is not None:
        save_root = opt.save_path
    else:
        mode_dir_name = os.path.dirname(opt.model_path)
        stime = mode_dir_name.split('\\')[-1]
        save_root = os.path.join(mode_dir_name, f'{stime}_results')

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

    if opt.multi_load:
        state_dict_multi = torch.load(opt.model_path)
        state_dict = {k[7:]: v for k, v in state_dict_multi.items()}
    else:
        state_dict = torch.load(opt.model_path)
    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    bce_loss = 0.0
    criterion = BCEDICE_loss
    tool_metric = ConfuseMatrixMeter(n_class=2)

    i = 0
    c_matrix = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}

    save_path = '/raid/SMOW-Net/output/'
    textfile = '/raid/SMOW-Net/datasets/GVLM-CD-256/list/test.txt'
    namelines = []
    with open(textfile, 'r', encoding='utf-8') as file:
        for c in file.readlines():
            namelines.append(c.strip('\n').split(' ')[0])

    with torch.no_grad():
        for reference, testimg, mask in tqdm.tqdm(test_loader):
            reference = reference.to(device).float()
            testimg = testimg.to(device).float()
            mask = mask.float()

            generated_mask = model(reference, testimg)
            if generated_mask.size(1) == 1:
                generated_mask = generated_mask.squeeze(1)
            else:
                generated_mask = torch.sigmoid(generated_mask)
                generated_mask = generated_mask[:, 1, :, :]
            generated_mask = generated_mask.to("cpu")
            
            bce_loss += criterion(generated_mask, mask)
            bin_genmask = (generated_mask > 0.5).numpy()
            bin_genmask = bin_genmask.astype(int).squeeze(0)
            mask = mask.numpy().astype(int).squeeze(0)
            color_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

            color_image[(bin_genmask == 1) & (mask == 1)] = [255, 255, 255]
            color_image[(bin_genmask == 0) & (mask == 0)] = [0, 0, 0]
            color_image[(bin_genmask == 1) & (mask == 0)] = [0, 0, 255]
            color_image[(bin_genmask == 0) & (mask == 1)] = [0, 255, 0]

            color_savename = save_path + namelines[i]
            cv2.imwrite(color_savename, color_image)

            i = i + 1

            tool_metric.update_cm(pr=bin_genmask, gt=mask)

        bce_loss /= len(test_loader)
        print("Test summary")
        print("Loss is {}".format(bce_loss))
        print()

        scores_dictionary = tool_metric.get_scores()
        print(scores_dictionary)

if __name__ == '__main__':
    main()