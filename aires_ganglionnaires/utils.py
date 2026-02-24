from nibabel.funcs import as_closest_canonical
import nibabel as nib
from nilearn.image import resample_to_img
import os
from typing import Optional, Union, Iterable
import torch


def load_as_closest_canonical(x, **kwargs):
    return as_closest_canonical(nib.load(x, **kwargs))


def load_resampled(
    reference_image,
    images_to_resample_paths: Optional[Union[str, Iterable]] = None,
    masks_to_resample_paths: Optional[Union[str, Iterable]] = None
):
    if isinstance(images_to_resample_paths, str):
        images_to_resample_paths = [images_to_resample_paths]
    if isinstance(masks_to_resample_paths, str):
        masks_to_resample_paths = [masks_to_resample_paths]
    if images_to_resample_paths is None:
        images_to_resample_paths = []
    if masks_to_resample_paths is None:
        masks_to_resample_paths = []
    images_to_resample = [
        resample_to_img(
            load_as_closest_canonical(path), reference_image)
        for path in images_to_resample_paths
        ]
    masks_to_resample = [
        resample_to_img(
            load_as_closest_canonical(path),
            reference_image,
            interpolation='nearest')
        for path in masks_to_resample_paths
        ]
    return images_to_resample, masks_to_resample


def img2torch(img, dtype=torch.float32):
    return torch.tensor(img.get_fdata()).to(dtype)
