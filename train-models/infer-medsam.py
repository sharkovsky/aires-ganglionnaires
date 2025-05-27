import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import sys

join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob
import SimpleITK as sitk
from tqdm import tqdm
import pandas as pd


work_dir = sys.argv[1]
outdir = f'/home/fcremonesi/aires-ganglionnaires/train-models/work_dir/{work_dir}'
MedSAM_CKPT_PATH = f"{outdir}/medsam_model_best_converted.pth"
use_boxes_from_csv = True
boxesqualifiername = '.totalsegboxes' if use_boxes_from_csv else ''

class MedSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks



device = "cpu"
sam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
medsam_model.eval()

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


def get_bbox_csv(patient, level, slice_):
    boxes_df = pd.read_csv('/home/fcremonesi/epione_storage/fcremonesi/data/rpa-dataset/Data/boxes.csv')
    box_np = boxes_df[(boxes_df.patient == patient) &
                      (boxes_df.level == level) &
                      (boxes_df.slice == slice_.item())].values[:,[-3,-4,-1,-2]].astype(np.int32)
    return box_np


def get_slices(patient, level):
    boxes_df = pd.read_csv('/home/fcremonesi/epione_storage/fcremonesi/data/rpa-dataset/Data/boxes.csv')
    return boxes_df[(boxes_df.patient == patient) &
                    (boxes_df.level == level)]['slice'].values.astype(np.int32)


def datafilename2maskfilename(fname):
    maskname = copy.copy(fname)
    maskname = maskname.replace('Data', 'mask')
    pat_id = fname.split('_')[-1].split('.')[0]
    maskname = '_'.join(maskname.split('_')[:-1])
    maskname += f'_y{pat_id}.nii.gz'
    return maskname


class RpaDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, data_list, window, level):
        self.root_path = root_path
        self.data_list = data_list
        self.l = level
        self.w = window

    def __getitem__(self, index):
        filename = os.path.join(self.root_path, self.data_list[index])
        img = sitk.ReadImage(filename)
        img = torch.permute(torch.tensor(sitk.GetArrayFromImage(img)), (1,2,0))
        img = torch.clip((img - self.l + 0.5*self.w)/self.w, 0, 1)
        filename = os.path.join(self.root_path, datafilename2maskfilename(self.data_list[index]))
        mask = sitk.ReadImage(filename)
        mask = torch.permute(torch.tensor(sitk.GetArrayFromImage(mask)), (1,2,0))
        return img, mask

    def __len__(self):
        return len(self.data_list)


path_to_aires_ganglionnaires = '/home/fcremonesi/epione_storage/fcremonesi/data/rpa-dataset/Data/'
data_list = [x for x in os.listdir(path_to_aires_ganglionnaires) if 'Data' in x]

rpadata = RpaDataset(path_to_aires_ganglionnaires, data_list, window=400, level=40)

for i in tqdm(range(len(rpadata))):
    if os.path.exists(os.path.join(outdir, rpadata.data_list[i].split('.nii.gz')[0] + f'{boxesqualifiername}.predmask.nii.gz')):
        print(f'Skipping {rpadata.data_list[i]}. File already exists.')
        continue
    img, mask = rpadata[i]
    W, H = img.shape[0], img.shape[1]
    print(rpadata.data_list[i])
    medsam_pred = torch.zeros_like(mask)
    for level, levelnum in levelname2num.items():
        binmask3d = (mask == levelnum).to(torch.uint8)
        #slices = torch.unique(torch.where(binmask3d > 0)[2])
        slices = get_slices(rpadata.data_list[i].split('.nii.gz')[0], level)
        for slice_ in tqdm(slices):
            binmask = binmask3d[...,slice_]
            imgslice = torch.repeat_interleave(img[:,:,slice_].unsqueeze(dim=0), 3, dim=0)
            imgslice = F.interpolate(imgslice[None, None, ...], size=(3,1024,1024)).squeeze(dim=(0,1))
            binmask = F.interpolate(binmask[None, None, ...], size=(1024,1024)).squeeze(dim=(0,1))
            if use_boxes_from_csv:
                rpaboxes = get_bbox_csv(rpadata.data_list[i].split('.nii.gz')[0], level, slice_)
                if rpaboxes.size == 0:
                    print(f"Could not compute Bounding Box for {rpadata.data_list[i]} {level} slice {slice_}")
                    continue
                rpaboxes = rpaboxes / np.array([W, H, W, H]) * 1024
            else:
                rpaboxes = get_bbox_2d(binmask, bbox_shift=0)
            with torch.no_grad():
                medsam_slice_logits = medsam_model(
                        imgslice.unsqueeze(dim=0).to(device),
                        torch.tensor(rpaboxes[np.newaxis,...]).to(device))
            medsam_slice_proba = F.sigmoid(F.interpolate(medsam_slice_logits, size=(512,512)).squeeze(dim=(0,1)))
            medsam_pred[:,:,slice_] = medsam_pred[:,:,slice_] + (1-(medsam_pred[:,:,slice_]>0).to(torch.uint8))*levelnum*(medsam_slice_proba.squeeze(dim=(0,1)).cpu() > 0.5)
    predimg = sitk.GetImageFromArray(medsam_pred.cpu().permute(2,0,1).numpy())
    predimg.CopyInformation(sitk.ReadImage(os.path.join(rpadata.root_path, rpadata.data_list[i])))
    sitk.WriteImage(predimg,
                    os.path.join(outdir,
                                 rpadata.data_list[i].split('.nii.gz')[0] + f'{boxesqualifiername}.predmask.nii.gz'))
    print(f"Finished writing {rpadata.data_list[i].split('.nii.gz')[0]}")
