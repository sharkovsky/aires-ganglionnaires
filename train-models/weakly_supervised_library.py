import os
import nibabel as nib
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
from torch.nn import BCELoss
from torch.nn.functional import relu
from torch import nan_to_num


class CAL_WSISDataset(Dataset):
    def __init__(self, 
                 image_path,
                 weak_labels_path,
                 patient_list,
                 target_level = 'level_ia_left',
                 transforms=None,
                 min_crop_shape=(32,32,64),
                 force_recompute_bbox=False,
                 only_crop_z=False,
                 crop_images=True,):
        self.image_path = image_path
        self.weak_labels_path = weak_labels_path
        self.patients = patient_list
        self.level = target_level
        self.transforms = transforms
        self.min_crop_shape = min_crop_shape
        self.force_recompute_bbox = force_recompute_bbox
        self.only_crop_z = only_crop_z
        self.crop_images = crop_images
    
    def __getitem__(self, idx):
        patient = self.patients[idx]
        img_path = os.path.join(self.image_path, f'{patient}.nii.gz')
        #print(img_path)
        label_path = os.path.join(self.weak_labels_path, patient, f'{self.level}.nii.gz')
        img = nib.load(img_path).get_fdata()
        label = nib.load(label_path).get_fdata()
        img, label = self._bbox_crop(img, label, patient, force_recompute=self.force_recompute_bbox)    
        if self.transforms is not None:
            img = self.transforms['data'](img)
        if self.transforms is not None:
            label = self.transforms['label'](label)
        return img, label
    
    def __len__(self):
        return len(self.patients)
    
    @staticmethod
    def get_bbox(label):
        w = np.where(label > 0)
        bbox_coords=[[],[]]
        for ax in range(3):
            lo,hi = w[ax].min(), w[ax].max()
            #print(f'{ax}, {lo}, {hi}')
            bbox_coords[0].append(lo)
            bbox_coords[1].append(hi)
        return bbox_coords
    
    @staticmethod
    def get_bbox_and_collate(label):
        bbox_coords = list()
        for b in range(label.shape[0]):
            bbox_coords.append(CAL_WSISDataset.get_bbox(label[b,0,...]))
        bbox_coords = torch.tensor(bbox_coords)
        return bbox_coords

    
    def _bbox_crop(self, img, label, patient, force_recompute=False):
        # Crop based on label
        bbox_coords_file = os.path.join(self.weak_labels_path, patient, f'{self.level}.bbox.csv')
        if os.path.exists(bbox_coords_file) and not force_recompute:
            bbox_coords = pd.read_csv(bbox_coords_file, header=0).values
        else:
            bbox_coords = self.get_bbox(label)
            bbox_coords = pd.DataFrame({
                '0': [bbox_coords[0][0],bbox_coords[1][0]], 
                '1': [bbox_coords[0][1],bbox_coords[1][1]], 
                '2': [bbox_coords[0][2],bbox_coords[1][2]]})
            bbox_coords.to_csv(bbox_coords_file, index=False)
            bbox_coords = bbox_coords.values
        if self.min_crop_shape is not None:
            # adjust cropping to avoid cropping smaller than crop_shape
            for ax in range(3):
                lo,hi = bbox_coords[0][ax], bbox_coords[1][ax]
                bbox_len = hi-lo
                if bbox_len < self.min_crop_shape[ax]:
                    diff = self.min_crop_shape[ax] - bbox_len
                    lo = lo - diff//2
                    hi = hi + diff//2 + diff%2
                    if lo < 0:
                        hi = hi - lo
                        lo = 0
                    elif hi >= img.shape[ax]:
                        lo = lo - (img.shape[ax] - hi - 1)
                        hi = img.shape[ax] - 1
                bbox_coords[0][ax] = lo
                bbox_coords[1][ax] = hi
        if self.only_crop_z and self.crop_images:
            img = img[:,:,bbox_coords[0][2]:bbox_coords[1][2]]
            label = label[:,:,bbox_coords[0][2]:bbox_coords[1][2]]
        elif self.crop_images:
            img = img[bbox_coords[0][0]:bbox_coords[1][0],bbox_coords[0][1]:bbox_coords[1][1],bbox_coords[0][2]:bbox_coords[1][2]]
            label = label[bbox_coords[0][0]:bbox_coords[1][0],bbox_coords[0][1]:bbox_coords[1][1],bbox_coords[0][2]:bbox_coords[1][2]]
        return img, label


##################  LOSSES  #################################
def _soft_threshold(x):
    return 2.*relu(x-0.5)
    

class WeakCELoss:
    def __init__(self):
        self.loss_fn = BCELoss(reduction='none')

    def __call__(self, pred, target):
        loss = self.loss_fn(pred, target)*(1 - target)
        loss = loss/((1 - target).sum(axis=(2,3,4))[...,None,None,None])
        return loss.sum(axis=(1,2,3,4)).squeeze()
    

class Log_barrier_extension:
    def __init__(self, t, mu):
        self.t= t
        self.mu = mu

    def __call__(self, z):
        mask = (z <= -1./(self.t*self.t)).type(torch.double)
        mask.requires_grad = False
        val = (-1./self.t)*nan_to_num(torch.log(-z))*mask + (self.t*z - 1/self.t*np.log(1./(self.t*self.t)) + 1./self.t)*(1-mask)
        return torch.clamp(val, min=-100., max=100.)

    def step(self):
        self.t *= self.mu


class Log_barrier_size_constraint_criterion:
    def __init__(self, lower_bound_factor=0.5, upper_bound_factor=None, t = 1., mu=1.):
        self.lower_bound = lower_bound_factor
        self.upper_bound = upper_bound_factor if upper_bound_factor is not None else 1.
        self.barrier = Log_barrier_extension(t, mu)

    def __call__(self, pred, target):
        pred_size = _soft_threshold(pred).sum(axis=(2,3,4))
        target_size = target.sum(axis=(2,3,4))
        #print(f'size: target {target_size} pred {pred_size}')
        loss = self.barrier(self.lower_bound - pred_size/target_size) 
        loss = loss + self.barrier(pred_size/target_size - self.upper_bound)
        return loss.squeeze()
    
    def step_barrier(self):
        self.barrier.step()


class Log_barrier_tightness_prior_criterion_testimpl:
    def __init__(self, t = 1., mu=1.):
        self.barrier = Log_barrier_extension(t, mu)

    def __call__(self, pred, target, bbox_coords):
        loss = torch.zeros(size=(pred.shape[0],1))
        with torch.no_grad():
            bbox_width = bbox_coords[:,1,:] - bbox_coords[:,0,:]
            w = bbox_width.sum(axis=1)
        for b in range(pred.shape[0]):
            s_ = pred[b,0,bbox_coords[b,0,0]:bbox_coords[b,1,0],bbox_coords[b,0,1]:bbox_coords[b,1,1],bbox_coords[b,0,2]:bbox_coords[b,1,2]].sum()
            loss[b,0] = loss[b,0] + self.barrier(w[b] - s_)
        return loss.squeeze()
    
    def step_barrier(self):
        self.barrier.step()
        
    
class Log_barrier_tightness_prior_criterion:
    def __init__(self, t = 1., mu=1.):
        self.barrier = Log_barrier_extension(t, mu)

    def __call__(self, pred, target, bbox_coords):
        loss = torch.zeros(size=(pred.shape[0],1))
        for b in range(pred.shape[0]):
            nlines = 0
            bbox_coords_ = bbox_coords[b,...]
            # over x
            for slice_ in range(bbox_coords_[0][0],bbox_coords_[1][0]+1):
                for line_ in range(bbox_coords_[0][2], bbox_coords_[1][2]):
                    loss[b,0] = loss[b,0] + self.barrier(1 - (pred[b,0,slice_,:,line_]).sum())
                    nlines += 1 
                for line_ in range(bbox_coords_[0][1], bbox_coords_[1][1]):
                    loss[b,0] = loss[b,0] + self.barrier(1 - (pred[b,0,slice_,line_,:]).sum())
                    nlines += 1
            #over y
            for slice_ in range(bbox_coords_[0][1],bbox_coords_[1][1]+1):
                for line_ in range(bbox_coords_[0][0], bbox_coords_[1][0]):
                    loss[b,0] = loss[b,0] + self.barrier(1 - (pred[b,0,line_,slice_,:]).sum())
                    nlines += 1
                for line_ in range(bbox_coords_[0][2], bbox_coords_[1][2]):
                    loss[b,0] = loss[b,0] + self.barrier(1 - (pred[b,0,:,slice_,line_]).sum())
                    nlines += 1
            # over z
            for slice_ in range(bbox_coords_[0][2],bbox_coords_[1][2]+1):
                for line_ in range(bbox_coords_[0][0], bbox_coords_[1][0]):
                    loss[b,0] = loss[b,0] + self.barrier(1 - (pred[b,0,line_,:,slice_]).sum())
                    nlines += 1
                for line_ in range(bbox_coords_[0][1], bbox_coords_[1][1]):
                    loss[b,0] = loss[b,0] + self.barrier(1 - (pred[b,0,:,line_,slice_]).sum())
                    nlines += 1
            loss[b,0] = loss[b,0]/(nlines+1e-11)
        return loss.squeeze()
    
    def step_barrier(self):
        self.barrier.step()


class Log_barrier_emptyness_constraint:
    def __init__(self,t = 1., mu=1.):
        self.barrier = Log_barrier_extension(t, mu)

    def __call__(self, pred, target, bbox_coords):
        loss = torch.zeros(size=(pred.shape[0],1))
        for b in range(pred.shape[0]):
            mask = torch.zeros_like(pred[b,...])
            mask[b,0,bbox_coords[b,0,0]:bbox_coords[b,1,0],bbox_coords[b,0,1]:bbox_coords[b,1,1],bbox_coords[b,0,2]:bbox_coords[b,1,2]] = 1.
            loss[b,0] = loss[b,0] + self.barrier(((pred*(1-mask)).sum(axis=(1,2,3,4)))/((1-mask).sum(axis=(1,2,3,4))[:,None,None,None,None]))
        return loss.squeeze()
    
    def step_barrier(self):
        self.barrier.step()


class Tightness_prior_argmax_impl:
    def __init__(self, epsilon=0.99):
        self.eps = epsilon
    def __call__(self, pred, target, bbox_coords):
        loss = torch.zeros(size=(pred.shape[0],1))
        for b in range(pred.shape[0]):
            nlines = 0
            bbox_coords_ = bbox_coords[b,...]
            # over x
            for slice_ in range(bbox_coords_[0][0],bbox_coords_[1][0]+1):
                for line_ in range(bbox_coords_[0][2], bbox_coords_[1][2]):
                    M = torch.max(pred[b,0,slice_,:,line_])
                    thresh = relu(pred[b,0,slice_,:,line_] - self.eps*M)/(1-self.eps)
                    loss[b,0] = loss[b,0] + (thresh*(1-target[b,0,slice_,:,line_])).sum()
                    nlines += 1
                for line_ in range(bbox_coords_[0][1], bbox_coords_[1][1]):
                    M = torch.max(pred[b,0,slice_,line_,:])
                    thresh = relu(pred[b,0,slice_,line_,:] - self.eps*M)/(1-self.eps)
                    loss[b,0] = loss[b,0] + (thresh*(1-target[b,0,slice_,line_,:])).sum()
                    nlines += 1
            #over y
            for slice_ in range(bbox_coords_[0][1],bbox_coords_[1][1]+1):
                for line_ in range(bbox_coords_[0][0], bbox_coords_[1][0]):
                    M = torch.max(pred[b,0,line_,slice_,:])
                    thresh = relu(pred[b,0,line_,slice_,:] - self.eps*M)/(1-self.eps)
                    loss[b,0] = loss[b,0] + (thresh*(1-target[b,0,line_,slice_,:])).sum()
                    nlines += 1
                for line_ in range(bbox_coords_[0][2], bbox_coords_[1][2]):
                    M = torch.max(pred[b,0,:,slice_,line_])
                    thresh = relu(pred[b,0,:,slice_,line_] - self.eps*M)/(1-self.eps)
                    loss[b,0] = loss[b,0] + (thresh*(1-target[b,0,:,slice_,line_])).sum()
                    nlines += 1
            # over z
            for slice_ in range(bbox_coords_[0][2],bbox_coords_[1][2]+1):
                for line_ in range(bbox_coords_[0][0], bbox_coords_[1][0]):
                    M = torch.max(pred[b,0,line_,:,slice_])
                    thresh = relu(pred[b,0,line_,:,slice_] - self.eps*M)/(1-self.eps)
                    loss[b,0] = loss[b,0] + (thresh*(1-target[b,0,line_,:,slice_])).sum()
                    nlines += 1
                for line_ in range(bbox_coords_[0][1], bbox_coords_[1][1]):
                    M = torch.max(pred[b,0,:,line_,slice_])
                    thresh = relu(pred[b,0,:,line_,slice_] - self.eps*M)/(1-self.eps)
                    loss[b,0] = loss[b,0] + (thresh*(1-target[b,0,:,line_,slice_])).sum()
                    nlines += 1
            loss[b,0] = loss[b,0]/(nlines+1e-11)
        return loss.squeeze()


class HomogeneousGradientRegularizer:
    def _gradient_from_batch(self, img_batch):
        grad = torch.zeros(size=(img_batch.shape[0], img_batch.shape[1], 3, img_batch.shape[2], img_batch.shape[3], img_batch.shape[4]))
        for ax in range(1,4):
            slice_ = [slice(None)] * 5
            slice_[-ax] = 0
            grad[:,:,-ax,...] = torch.diff(img_batch,prepend=img_batch[slice_].unsqueeze(dim=-ax),dim=-ax)
        return grad

    def __call__(self, pred, target, img, return_grad_norm=False):
        thresh = _soft_threshold(pred)
        with torch.no_grad():
            grad = self._gradient_from_batch(img)
        grad_norm = (grad*grad*thresh).sum(axis=2)
        grad_mean = grad_norm.sum(axis=(1,2,3,4))  #/thresh.sum(axis=(1,2,3,4))  # experiment without mean
        if return_grad_norm:
            return grad_mean, grad_norm
        else:
            return grad_mean
        

class AverageWindowedHounsfieldRegularizer:
    def __init__(self, target_houns, window, level, std_target_houns):
        self.target_val = (target_houns - level + 0.5*window)/window
        if self.target_val > 1:
            self.target_val = 1
        elif self.target_val < 0:
            self.target_val = 0.
        self.lambda_ = std_target_houns

    def __call__(self,pred,img):
        thresh = _soft_threshold(pred)
        avg_houns = (thresh*img).sum(axis=(1,2,3,4)).sum()/(1e-6+thresh.sum(axis=(1,2,3,4)))
        return (avg_houns - self.target_val)*(avg_houns - self.target_val)/(self.lambda_*self.lambda_)


class WindowedHounsfieldRegularizer:
    def __init__(self, target_houns, window, level, std_target_houns):
        self.target_val = (target_houns - level + 0.5*window)/window
        if self.target_val > 1:
            self.target_val = 1
        elif self.target_val < 0:
            self.target_val = 0.
        self.lambda_ = std_target_houns

    def __call__(self,pred,img):
        thresh = _soft_threshold(pred)
        diff = thresh*img
        diff = (diff - self.target_val)*(diff - self.target_val)/(self.lambda_*self.lambda_)
        return diff.sum(axis=(1,2,3,4)).sum()/(1e-6+thresh.sum(axis=(1,2,3,4))) 
    

class PredictedGradientRegularizer:
    def _gradient_from_batch(self, img_batch):
        grad = torch.zeros(size=(img_batch.shape[0], img_batch.shape[1], 3, img_batch.shape[2], img_batch.shape[3], img_batch.shape[4]))
        for ax in range(1,4):
            slice_ = [slice(None)] * 5
            slice_[-ax] = 0
            grad[:,:,-ax,...] = torch.diff(img_batch,prepend=img_batch[slice_].unsqueeze(dim=-ax),dim=-ax)
        return grad

    def __call__(self, pred, return_grad_norm=False):
        grad = self._gradient_from_batch(pred)
        grad_norm = (grad*grad).sum(axis=2)
        grad_mean = grad_norm.mean(axis=(1,2,3,4))
        if return_grad_norm:
            return grad_mean, grad_norm
        else:
            return grad_mean
        

class ACMWEReg:
    def __init__(self,lambda_):
        self.lambda_ = lambda_

    def _gradient_from_batch(self, img_batch):
        grad = torch.zeros(size=(img_batch.shape[0], img_batch.shape[1], 3, img_batch.shape[2], img_batch.shape[3], img_batch.shape[4]))
        for ax in range(1,4):
            slice_ = [slice(None)] * 5
            slice_[-ax] = 0
            grad[:,:,-ax,...] = torch.diff(img_batch,prepend=img_batch[slice_].unsqueeze(dim=-ax),dim=-ax)
        return grad
    
    def __call__(self, pred, img):
        gradu = self._gradient_from_batch(pred)
        grad_norm = torch.sqrt((gradu*gradu).sum(axis=2) + 1e-6)
        length = grad_norm.sum(axis=(1,2,3,4))
        thresh = _soft_threshold(pred)
        c1 = (thresh*img).sum(axis=(1,2,3,4))/(1e-6+thresh.sum(axis=(1,2,3,4)))
        c2 = ((1-thresh)*img).sum(axis=(1,2,3,4))/(1e-6+(1-thresh).sum(axis=(1,2,3,4)))
        region = (thresh*(c1 - img)*(c1 - img)).sum(axis=(1,2,3,4))
        region = region + ((1-thresh)*(c2 - img)*(c2 - img)).sum(axis=(1,2,3,4))
        #print(length, region, c1, c2)
        return length + self.lambda_*region

