# %% setup environment
import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import monai
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob
import pandas as pd
import SimpleITK as sitk
import copy

# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

num2levelname = {
     1: 'level_ia_left',
     2: 'level_ia_right',
     3: 'level_ib_left',
     4: 'level_ib_right',
     5: 'level_ii_left',
     6: 'level_ii_right',
     7: 'level_iii_left',
     8: 'level_iii_right',
     9: 'level_iv_left',
    10: 'level_iv_right',
    11: 'level_v_left',
    12: 'level_v_right',
#    13: 'level_rp_left',
#    14: 'level_rp_right',
}
levelname2num = {val: key for key, val in num2levelname.items()}

def get_bbox_2d(gt2D, bbox_shift=5):
    y_indices, x_indices = np.where(gt2D > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = gt2D.shape
    x_min = max(0, x_min - random.randint(0, bbox_shift))
    x_max = min(W, x_max + random.randint(0, bbox_shift))
    y_min = max(0, y_min - random.randint(0, bbox_shift))
    y_max = min(H, y_max + random.randint(0, bbox_shift))
    bboxes = np.array([x_min, y_min, x_max, y_max])
    return bboxes


def datafilename2maskfilename(fname):
    maskname = copy.copy(fname)
    maskname = maskname.replace('Data', 'mask')
    pat_id = fname.split('_')[-1].split('.')[0]
    maskname = '_'.join(maskname.split('_')[:-1])
    maskname += f'_y{pat_id}.nii.gz'
    return maskname


def masknumbers2channels(mask):
    num_levels = len(num2levelname)
    out = torch.stack(num_levels*[torch.zeros_like(mask)], dim=0)
    for levelnum in num2levelname.keys():
        out[levelnum-1,...] = (mask == levelnum)
    return out.to(torch.uint8)


def channels2masknumbers(mask):
    out = torch.zeros(size=mask.shape[1:])
    out = out.to(torch.uint8)
    out = out.to(device)
    for levelnum in num2levelname.keys():
        out += (out == 0)*mask[levelnum-1,...].to(torch.uint8)*levelnum
    return out.to(torch.uint8)


class RpaDatasetForInference(torch.utils.data.Dataset):
    def __init__(self, root_path, output_path, data_list, window, level):
        self.root_path = root_path
        self.output_path = output_path
        self.data_list = data_list
        self.l = level
        self.w = window
        self.boxes_df = pd.read_csv('/home/fcremonesi/epione_storage/fcremonesi/data/rpa-dataset/Data/boxes.csv')
        self.xy_len = 256
        self.z_len = 128

    def __getitem__(self, index):
        # Read image
        filename = os.path.join(self.root_path, self.data_list[index])
        img = sitk.ReadImage(filename)
        img = torch.permute(torch.tensor(sitk.GetArrayFromImage(img)), (1,2,0))
        W, H, D = img.shape[0], img.shape[1], img.shape[2]
        # Apply windowing
        img = torch.clip((img - self.l + 0.5*self.w)/self.w, 0, 1)
        # Read mask
        filename = os.path.join(self.root_path, datafilename2maskfilename(self.data_list[index]))
        mask = sitk.ReadImage(filename)
        metadata = {'shape': {'image': img.shape}, 'mask': mask, 'name': self.data_list[index]}
        mask = torch.permute(torch.tensor(sitk.GetArrayFromImage(mask)), (1,2,0))
        metadata['shape']['mask'] = mask.shape
        mask = masknumbers2channels(mask)

        img = F.interpolate(img.unsqueeze(dim=0).unsqueeze(dim=0), size=(self.xy_len, self.xy_len, self.z_len)).squeeze(dim=0)
        binmask = F.interpolate(mask.unsqueeze(dim=0), size=(self.xy_len, self.xy_len, self.z_len)).squeeze()
        return img, binmask, metadata

    def postprocess(self, prediction, metadata):
        prediction = (F.sigmoid(prediction) > 0.5).to(torch.uint8) 
        prediction = channels2masknumbers(prediction)
        return F.interpolate(prediction.unsqueeze(dim=0).unsqueeze(dim=0), size=(metadata['shape']['mask'])).squeeze(dim=(0,1))
    
    def save_inferred_mask(self, mask, metadata):
        maskimg = sitk.GetImageFromArray(torch.permute(mask.squeeze(dim=0).cpu(), (2, 0, 1)))
        maskimg.CopyInformation(metadata['mask'])
        filename = metadata['name'].split('.')[0] + '_prediction.nii.gz'
        print(f'Saving {self.output_path}/{filename}')
        sitk.WriteImage(maskimg, f'{self.output_path}/{filename}')

    def __len__(self):
        return len(self.data_list)

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--tr_npy_path",
    type=str,
    default="data/npy/CT_Abd",
    help="path to training npy files; two subfolders: gts and imgs",
)
parser.add_argument("-task_name", type=str, default="UNet")
parser.add_argument("-model_type", type=str, default="vit_b")
parser.add_argument("-work_dir", type=str, default="./work_dir")
# train
parser.add_argument("-num_epochs", type=int, default=1000)
parser.add_argument("-batch_size", type=int, default=2)
parser.add_argument("-num_workers", type=int, default=0)
# Optimizer parameters
parser.add_argument(
    "-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)"
)
parser.add_argument(
    "-lr", type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)"
)
parser.add_argument(
    "--resume", type=str, default="", help="Resuming training from checkpoint"
)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--eval_fold", type=int, default=0)
args = parser.parse_args()


# %% set up model for training
# device = args.device
run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
device = torch.device(args.device)
# %% set up model
unet_model = monai.networks.nets.UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=len(num2levelname),
        channels=(64,128,256,512,1024),
        strides=(2,2,2,2)
        )


eval_folds = [
#        [10,11,12,13,14,44],
#        [10,11,12,13,14,44,25,2,45,21,33,37,31,8,20,1,6,18,22,9,42,30,32,23,3,38],
        [4, 18, 35, 1, 27],
        [16, 5, 41, 40, 32], 
        [33, 26, 7, 30, 14], 
        [6, 37, 9, 23, 31], 
        [28, 17, 2, 12, 45], 
        [8, 43, 29, 38, 20], 
        [22, 10, 36, 24, 13], 
        [39, 19, 34, 15, 21], 
        [3, 11, 42, 44, 25],
        ]

unet_model.to(device)

def main():
    print(
        "Number of total parameters: ",
        sum(p.numel() for p in unet_model.parameters()),
    )  # 93735472

    data_list = [x for x in os.listdir(args.tr_npy_path) if 'Data' in x]
    print('Inference patients: ', data_list)
    train_dataset = RpaDatasetForInference(
            args.tr_npy_path,
            os.path.join(os.path.split(args.resume)[0]),
            data_list,
            window=400,
            level=40)

    print("Number of inference samples: ", len(train_dataset))

    if args.resume is not None:
        if os.path.isfile(args.resume):
            ## Map model to be loaded to specified single GPU
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            unet_model.load_state_dict(checkpoint["model"])
    else:
        assert False, "Need a model to resume!"

    unet_model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(train_dataset))):
            image, gt2D, metadata = train_dataset[i]
            image, gt2D = image.to(device), gt2D.to(device)
            unet_pred = unet_model(image.unsqueeze(dim=0)).squeeze(dim=(0,1))
            mask = train_dataset.postprocess(unet_pred, metadata)
            train_dataset.save_inferred_mask(mask, metadata)


if __name__ == "__main__":
    main()
