import numpy as np
import nibabel as nib
from skimage.morphology import label
import os
from typing import List, Dict, Optional
from functools import reduce


totalseg_tasks = ['total', 'tissue', 'head_muscles', 'head_glands_cavities', 'headneck_bones_vessels', 'headneck_muscles']


specs_to_args = {
    'inferior border': {'axis': 2, 'get_largest_index': False,  'one_after': True},
    'superior border': {'axis': 2, 'get_largest_index': True,  'one_after': False},
    'anterior border': {'axis': 1, 'get_largest_index': True,  'one_after': False},
    'posterior border': {'axis': 1, 'get_largest_index': False,  'one_after': True},
    'left border': {'axis': 0, 'get_largest_index': True,  'one_after': False},
    'right border': {'axis': 0, 'get_largest_index': False,  'one_after': True},
}


def get_extremal_idx(organ_segmentation: 'ImageSegmentation', axis: int, get_largest_index: bool = True) -> int:
    """Gets the index corresponding to the first or last nonzero pixel on an axis.
    
    
    Args:
        organ_segmentation: the binary mask representing the segmentation of an organ
        axis: the integer corresponding to the axis where we want to find the extreme point (e.g. 2 for z axis)
        get_largest_index: boolean to get the largest or the smallest index on the projection

    Examples:
        for the z axis, the position of the highest pixel in the organ segmentation can be found with
        ```
        get_extremal_idx(organ_segmentation, 2, get_largest_index=True)
        ```

        while the following finds the position of the lowest pixel in the organ segmentation
        ```
        get_extremal_idx(organ_segmentation, 2, get_largest_index=False)
        ```

    Returns:
        pixel_index: the integer index corresponding to the last (resp. first) nonzero pixel on the axis

    """
    all_ax = [0,1,2]
    all_ax.pop(axis)
    projection = organ_segmentation
    for ax_ in sorted(all_ax)[::-1]:
        projection = projection.sum(ax_)
    pixel_index = np.max(np.where(projection > 0)[0]) if get_largest_index else np.min(np.where(projection > 0)[0])
    return pixel_index


def get_extremal_idx_by_z_slice(organ_segmentation: 'ImageSegmentation', axis: int, get_largest_index: bool = True) -> List[int]:
    projection = organ_segmentation.sum(1-axis)
    axis_indices = list()
    z_indices = np.unique(np.where(projection > 0)[1])
    for z in z_indices:
        axis_indices.append(
            np.max(np.where(projection[...,z] > 0)[0]) if get_largest_index else np.min(np.where(projection[...,z] > 0)[0])
        )
    return np.array(axis_indices), z_indices


def define_area_by_plane(organ_segmentation: 'ImageSegmentation',
                         axis: int,
                         get_largest_index: bool = True,
                         one_after: bool = True,
                         slice_by_slice: bool = False) -> 'ImageSegmentation':
    """Returns a new mask that defines a 3d-rectangular area in the image.

    The new mask is split in two areas, divided by the plane perpendicular to `axis` at the position 
    defined by `get_extremal_idx(mask, axis, get_largest_index)`.
    
    If `one_after` is True, all pixels with a larger index than the plane are set to 1, all the
    others are set to 0. If `one_after` is False, the opposite happens.
    
    Args:
        organ_segmentation: the binary mask representing the segmentation of an organ
        axis: the integer corresponding to the axis where we want to find the extreme point (e.g. 2 for z axis)
        get_largest_index: boolean to get the largest or the smallest index on the projection
        one_after: boolean as explained above
        slice_by_slice: boolean to toggle processing each z-slice separately (True) or combined (False, default)

    Returns:
        area_mask: a binary image with the same shape as mask 

    """
    if slice_by_slice:
        area_mask = np.zeros_like(organ_segmentation)
        axis_indices, z_indices = get_extremal_idx_by_z_slice(organ_segmentation, axis, get_largest_index)
        for i in range(z_indices):
            selecting_slice = slice(axis_indices[i], None) if one_after else slice(None, axis_indices[i])
            match axis:
                case 0:
                    indices = (selecting_slice, slice(None), z_indices[i])
                case 1:
                    indices = (slice(None), selecting_slice, z_indices[i])
            area_mask[indices] = 1
    else:
        idx = get_extremal_idx(organ_segmentation, axis, get_largest_index)
        area_mask = np.zeros_like(organ_segmentation)
        selecting_slice = slice(idx, None) if one_after else slice(None, idx)
        match axis:
            case 0:
                indices = (selecting_slice, Ellipsis)
            case 1:
                indices = (slice(None), selecting_slice, slice(None))
            case 2:
                indices = (Ellipsis, selecting_slice)
        area_mask[indices] = 1
    return area_mask


def define_area_by_specs(area_specs: Dict[str, List[Dict]],
        patient: str,
        path_to_totalseg_segmentations: str,
        totalseg_structure_to_task) -> 'ImageSegmentation':
    """Returns a new mask that defines an aire ganglionnaire based on a dictionary of specifications.

    The specifications explain how each border of the aire ganglionnaire is defined. 
    The format of the specs is:
    ```
    {border of the aire ganglionnaire: [structure_specs]}
    ```
    Each `structure_spec` is itself a dictionary with the keys:
    - structure: totalsegmentator filename
    - border: which border of the totalsegmentator class defines the border of the aire ganglionnaire

    Example:
    ```
    level_iia_left_specs = {
        'inferior border': [{'border': ['inferior border'], 'structure': 'hyoid'}],
        'superior border': [{'border': ['superior border'], 'structure': 'vertebrae_C1'}],
        'posterior border': [{'border': ['posterior border'], 'structure': 'internal_jugular_vein_left'}],
        'anterior border': [{'border': ['posterior border'], 'structure': 'submandibular_gland_left'}],
        'left border': [{'border': ['left border'], 'structure': 'sternocleidomastoid_left'}],
        'right border': [{'border': ['right border'], 'structure': 'internal_carotid_artery_left'}],
    }
```
"""
    def _gen_areas():
        for border, specs in area_specs.items():
            one_after_ = specs_to_args[border]['one_after']
            axis_ = specs_to_args[border]['axis']
            for organ_border_specs in specs:
                seg_filename = organ_border_specs['structure'] + '.nii.gz'
                for organ_border_name in organ_border_specs['border']:
                    get_largest_index_ = specs_to_args[organ_border_name]['get_largest_index']
                    seg_file_path = os.path.join(
                        path_to_totalseg_segmentations,
                        totalseg_structure_to_task[seg_filename],
                        patient,
                        seg_filename
                    )
                    organ_segmentation = nib.load(seg_file_path).get_fdata()
                yield define_area_by_plane(organ_segmentation, axis_,get_largest_index_, one_after_)
    return reduce(lambda x,y: x*y, _gen_areas())


def define_area_by_specs_with_heuristics(area_specs: Dict[str, List[Dict]],
        patient: str,
        path_to_totalseg_segmentations: str,
        totalseg_structure_to_task) -> 'ImageSegmentation':
    """Returns a new mask that defines an aire ganglionnaire based on a dictionary of specifications.

    Implements some heuristics:
    - first define superior and inferior borders
    - for x and y borders, only search for extrema *within the top and bottom slice*
    - for x and y borders, search for extrema slice-by-slice instead of globally

    The specifications explain how each border of the aire ganglionnaire is defined. 
    The format of the specs is:
    ```
    {border of the aire ganglionnaire: [structure_specs]}
    ```
    Each `structure_spec` is itself a dictionary with the keys:
    - structure: totalsegmentator filename
    - border: which border of the totalsegmentator class defines the border of the aire ganglionnaire

    Example:
    ```
    level_iia_left_specs = {
        'inferior border': [{'border': ['inferior border'], 'structure': 'hyoid'}],
        'superior border': [{'border': ['superior border'], 'structure': 'vertebrae_C1'}],
        'posterior border': [{'border': ['posterior border'], 'structure': 'internal_jugular_vein_left'}],
        'anterior border': [{'border': ['posterior border'], 'structure': 'submandibular_gland_left'}],
        'left border': [{'border': ['left border'], 'structure': 'sternocleidomastoid_left'}],
        'right border': [{'border': ['right border'], 'structure': 'internal_carotid_artery_left'}],
    }
```
"""
    do_first_borders =  ['superior border', 'inferior border']
    slice_by_slice_ = True
    def _gen_areas():
        for border in do_first_borders:
            specs = area_specs[border]
            one_after_ = specs_to_args[border]['one_after']
            axis_ = specs_to_args[border]['axis']
            for organ_border_specs in specs:
                seg_filename = organ_border_specs['structure'] + '.nii.gz'
                for organ_border_name in organ_border_specs['border']:
                    get_largest_index_ = specs_to_args[organ_border_name]['get_largest_index']
                    seg_file_path = os.path.join(
                        path_to_totalseg_segmentations,
                        totalseg_structure_to_task[seg_filename],
                        patient,
                        seg_filename
                    )
                    organ_segmentation = nib.load(seg_file_path).get_fdata()
                yield define_area_by_plane(organ_segmentation, axis_, get_largest_index_, one_after_, slice_by_slice_)
 
        # do the other borders
        for border, specs in area_specs.items():
            if border in do_first_borders:
                continue
            one_after_ = specs_to_args[border]['one_after']
            axis_ = specs_to_args[border]['axis']
            for organ_border_specs in specs:
                seg_filename = organ_border_specs['structure'] + '.nii.gz'
                for organ_border_name in organ_border_specs['border']:
                    get_largest_index_ = specs_to_args[organ_border_name]['get_largest_index']
                    seg_file_path = os.path.join(
                        path_to_totalseg_segmentations,
                        totalseg_structure_to_task[seg_filename],
                        patient,
                        seg_filename
                    )
                    organ_segmentation = nib.load(seg_file_path).get_fdata()
                yield define_area_by_plane(organ_segmentation, axis_, get_largest_index_, one_after_, slice_by_slice_)
    return reduce(lambda x,y: x*y, _gen_areas())


def extract_largest_connected_component(mask):
    label_img  = label(mask)
    max_vol = 0
    max_label = -1
    for lab_ in range(1,label_img.max()+1):
        vol_ = (label_img == lab_).sum()
        if vol_ > max_vol:
            max_label = lab_
            max_vol = vol_
    return (label_img == max_label).astype(np.int32)

