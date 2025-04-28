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

class RpaDatasetForTraining(torch.utils.data.Dataset):
    def __init__(self, root_path, data_list, window, level):
        self.root_path = root_path
        self.data_list = data_list
        self.l = level
        self.w = window
        self.boxes_df = pd.read_csv('/home/fcremonesi/epione_storage/fcremonesi/data/rpa-dataset/Data/boxes.csv')

    def __getitem__(self, index):
        # Read image
        filename = os.path.join(self.root_path, self.data_list[index])
        img = sitk.ReadImage(filename)
        img = torch.permute(torch.tensor(sitk.GetArrayFromImage(img)), (1,2,0))
        W, H = img.shape[0], img.shape[1]
        # Apply windowing
        img = torch.clip((img - self.l + 0.5*self.w)/self.w, 0, 1)
        # Read mask
        filename = os.path.join(self.root_path, datafilename2maskfilename(self.data_list[index]))
        mask = sitk.ReadImage(filename)
        mask = torch.permute(torch.tensor(sitk.GetArrayFromImage(mask)), (1,2,0))
        # Get bounding box
        patient = self.data_list[index].split('.nii.gz')[0]
        patientboxes_df = self.boxes_df[self.boxes_df.patient == patient]
        level = np.random.choice(patientboxes_df.level.unique())
        df = patientboxes_df[patientboxes_df.level == level]
        slice_ = np.random.choice(df.slice.unique())
        box_np = df[df.slice == slice_].values[:,[-3,-4,-1,-2]].astype(np.int32)
        # Slice tensors
        levelnum = levelname2num[level]
        binmask3d = (mask == levelnum).to(torch.uint8)
        binmask = binmask3d[...,slice_]
        # Resize to 1024x1024
        #imgslice = torch.repeat_interleave(img[:,:,slice_].unsqueeze(dim=0), 3, dim=0)  # for UNet keep a single channel
        imgslice = img[:,:,slice_].unsqueeze(dim=0)
        imgslice = F.interpolate(imgslice[None, None, ...], size=(1,1024,1024)).squeeze(dim=(0,1))
        binmask = F.interpolate(binmask[None, None, ...], size=(1024,1024)).squeeze(dim=(0,1))
        box_np = box_np / np.array([W, H, W, H]) * 1024
        return imgslice, binmask, torch.tensor(box_np)

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
args = parser.parse_args()

# %% set up model for training
# device = args.device
run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
device = torch.device(args.device)

# %% set up model
unet_model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(32,64,128),
        strides=(2,2)
        )

unet_model.to(device)

def main():
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )

    print(
        "Number of total parameters: ",
        sum(p.numel() for p in unet_model.parameters()),
    )  # 93735472
    print(
        "Number of trainable parameters: ",
        sum(p.numel() for p in unet_model.parameters() if p.requires_grad),
    )  # 93729252

    optimizer = torch.optim.AdamW(
        unet_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    num_epochs = args.num_epochs
    iter_num = 0
    losses = []
    best_loss = 1e10
    eval_patients = [10,11,12,13,14,44]
    eval_data_list = [f'Overall_Data_Examples_{patid}.nii.gz' for patid in eval_patients]
    data_list = [x for x in os.listdir(args.tr_npy_path) if 'Data' in x and x not in eval_data_list]
    print('Training patients: ', data_list)
    train_dataset = RpaDatasetForTraining(
            args.tr_npy_path,
            data_list,
            window=400,
            level=40)


    print("Number of training samples: ", len(train_dataset))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f'LR {args.lr}')

    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            ## Map model to be loaded to specified single GPU
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            unet_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        for step, (image, gt2D, boxes) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            boxes_np = boxes.detach().cpu().numpy()
            image, gt2D = image.to(device), gt2D.to(device)
            unet_pred = unet_model(image).squeeze(dim=1)
            loss = seg_loss(unet_pred, gt2D) + ce_loss(unet_pred, gt2D.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            iter_num += 1

        epoch_loss /= step
        losses.append(epoch_loss)

        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
        )
        ## save the latest model
        checkpoint = {
            "model": unet_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, join(model_save_path, "unet_model_latest.pth"))

        ## save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint = {
                "model": unet_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, "unet_model_best.pth"))

        # %% plot loss
        plt.plot(losses)
        plt.title("Dice + Cross Entropy Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(join(model_save_path, args.task_name + "train_loss.png"))
        plt.close()


if __name__ == "__main__":
    main()
