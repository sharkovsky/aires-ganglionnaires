import numpy as np
import nibabel as nib
import torch
from skimage.morphology import label
import os
from typing import List, Dict
from functools import reduce


_area_border_specs_to_args = {  # these are the target area's borders
    'inferior border': {'axis': 2, 'one_after': True},
    'superior border': {'axis': 2, 'one_after': False},
    'anterior border': {'axis': 1, 'one_after': False},
    'posterior border': {'axis': 1, 'one_after': True},
    'left border': {'axis': 0, 'one_after': True},
    'right border': {'axis': 0, 'one_after': False},
}

_structure_border_specs_to_args = {  # these are the anchor structure's borders
    'inferior border': {'get_largest_index': False, 'slice_by_slice': False, 'line_by_line': True},
    'superior border': {'get_largest_index': True, 'slice_by_slice': False, 'line_by_line': True},
    'anterior border': {'get_largest_index': True, 'slice_by_slice': False, 'line_by_line': True},
    'posterior border': {'get_largest_index': False, 'slice_by_slice': False, 'line_by_line': True},
    'left border': {'get_largest_index': False, 'slice_by_slice': False, 'line_by_line': True},
    'right border': {'get_largest_index': True, 'slice_by_slice': False, 'line_by_line': True},
    'inferior point': {'get_largest_index': False, 'slice_by_slice': False, 'line_by_line': False},
    'superior point': {'get_largest_index': True, 'slice_by_slice': False, 'line_by_line': False},
    'anterior point': {'get_largest_index': True, 'slice_by_slice': True, 'line_by_line': False},
    'posterior point': {'get_largest_index': False, 'slice_by_slice': True, 'line_by_line': False},
    'left point': {'get_largest_index': False, 'slice_by_slice': True, 'line_by_line': False},
    'right point': {'get_largest_index': True, 'slice_by_slice': True, 'line_by_line': False},
}


def _get_extremal_idx(organ_segmentation: 'ImageSegmentation',
                     axis: int,
                     get_largest_index: bool = True) -> int:
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
    all_ax = [0, 1, 2]
    all_ax.pop(axis)
    projection = organ_segmentation
    for ax_ in sorted(all_ax)[::-1]:
        projection = projection.sum(ax_)
    pixel_index = np.max(np.where(projection > 0)[0]) if get_largest_index else np.min(np.where(projection > 0)[0])
    return pixel_index


def _get_extremal_idx_by_z_slice(organ_segmentation: 'ImageSegmentation', axis: int, get_largest_index: bool = True) -> List[int]:
    projection = organ_segmentation.sum(1-axis)
    axis_indices = list()
    z_indices = np.unique(np.where(projection > 0)[1])
    for z in z_indices:
        axis_indices.append(
            np.max(np.where(projection[...,z] > 0)[0]) if get_largest_index else np.min(np.where(projection[...,z] > 0)[0])
        )
    return np.array(axis_indices), z_indices


def _get_extremal_idx_line_by_line(organ_segmentation: 'ImageSegmentation', axis: int, get_largest_index: bool = True) -> List[int]:
    indices = [[], [], []]
    w = np.array(np.where(organ_segmentation > 0))
    other_ax = [0, 1, 2]
    _ = other_ax.pop(axis)
    ax1 = other_ax[-1]
    ax2 = other_ax[0]
    for ax1_idx in np.unique(w[ax1, :]):
        for ax2_idx in np.unique(w[ax2, w[ax1, :] == ax1_idx]):
            axis_idx = np.max(w[axis, (w[ax1, :] == ax1_idx) & (w[ax2, :] == ax2_idx)]) if get_largest_index else np.min(w[axis, (w[ax1, :] == ax1_idx) & (w[ax2, :] == ax2_idx)])
            indices[ax1].append(int(ax1_idx))
            indices[ax2].append(int(ax2_idx))
            indices[axis].append(int(axis_idx))
    return list(map(lambda x: np.array(x), indices))
    

def define_area_by_plane(organ_segmentation: 'ImageSegmentation',
                         axis: int,
                         get_largest_index: bool = True,
                         one_after: bool = True,
                         slice_by_slice: bool = False,
                         line_by_line: bool = False) -> 'ImageSegmentation':
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
    area_mask = torch.zeros_like(organ_segmentation)
    if line_by_line:
        axis_indices = _get_extremal_idx_line_by_line(organ_segmentation, axis, get_largest_index)
        other_ax = [0, 1, 2]
        _ = other_ax.pop(axis)
        ax1 = other_ax[-1]  # always z dimension
        ax2 = other_ax[0]
        # slice by slice
        bbox_slices_min, bbox_slices_max = axis_indices[ax1].min(), axis_indices[ax1].max()
        for slice_ in reversed(range(bbox_slices_min, bbox_slices_max+1)):
            w = np.where(axis_indices[ax1] == slice_)[0]
            if len(w) == 0:
                # slice was not segmented: replicate previous slice
                selecting_slice = [slice(None),slice(None),slice(None)]
                selecting_slice[ax1] = slice_ - 1
                selecting_slice[ax2] = slice(None)
                selecting_slice[axis] = slice(None)
                setting_slice = [slice(None),slice(None),slice(None)]
                setting_slice[ax1] = slice_
                setting_slice[ax2] = slice(None)
                setting_slice[axis] = slice(None)
                area_mask[*setting_slice] = area_mask[*selecting_slice]
                continue
            # from 0 until min ax2
            minax2idx = np.argmin(axis_indices[ax2][w])
            selecting_slice = [slice(None),slice(None),slice(None)]
            selecting_slice[ax1] = slice_
            selecting_slice[ax2] = slice(0,axis_indices[ax2][w[minax2idx]])
            selecting_slice[axis] = slice(axis_indices[axis][w[minax2idx]], None) if one_after else slice(None, axis_indices[axis][w[minax2idx]])
            area_mask[*selecting_slice] = 1
            # inside the structure, i.e. between min ax2 and max ax2
            maxax2idx = np.argmax(axis_indices[ax2][w])
            for ax2_coord in range(axis_indices[ax2][w[minax2idx]],axis_indices[ax2][w[maxax2idx]]):
                selecting_slice = [slice(None),slice(None),slice(None)]
                selecting_slice[ax1] = slice_
                selecting_slice[ax2] = ax2_coord
                try:
                    axis_idx = np.where(axis_indices[ax2][w] == ax2_coord)[0][0]
                except IndexError:
                    # ignore and use previous value of axis_idx
                    pass
                selecting_slice[axis] = slice(axis_indices[axis][w[axis_idx]], None) if one_after else slice(None, axis_indices[axis][w[axis_idx]])
                area_mask[*selecting_slice] = 1
            # from max ax2 until the end
            selecting_slice = [slice(None),slice(None),slice(None)]
            selecting_slice[ax1] = slice_
            selecting_slice[ax2] = slice(axis_indices[ax2][w[maxax2idx]], None)
            selecting_slice[axis] = slice(axis_indices[axis][w[maxax2idx]], None) if one_after else slice(None, axis_indices[axis][w[maxax2idx]])
            area_mask[*selecting_slice] = 1
        # extensions
        # note: extensions work making some specific assumptions:
        # the behaviour for z-direction extension (which is always ax1) is different than the left-right or anterior-posterior extension
        # extend under min ax1: base case
        ax1_idx = np.min(axis_indices[ax1])
        w = np.where(axis_indices[ax1] == ax1_idx)[0]
        for i in w:
            selecting_slice = [slice(None),slice(None),slice(None)]
            selecting_slice[ax1] = slice(0,ax1_idx)
            selecting_slice[ax2] = axis_indices[ax2][i]
            selecting_slice[axis] = slice(axis_indices[axis][i], None) if one_after else slice(None, axis_indices[axis][i])
            area_mask[*selecting_slice] = 1
        # extend under min ax1: ax2 extensions
        # min ax2
        minax2idx = np.argmin(axis_indices[ax2][w])
        selecting_slice = [slice(None),slice(None),slice(None)]
        selecting_slice[ax1] = slice(0,ax1_idx)
        selecting_slice[ax2] = slice(0,axis_indices[ax2][w[minax2idx]])
        selecting_slice[axis] = slice(axis_indices[axis][w[minax2idx]], None) if one_after else slice(None, axis_indices[axis][w[minax2idx]])
        area_mask[*selecting_slice] = 1
        # max ax2
        maxax2idx = np.argmax(axis_indices[ax2][w])
        selecting_slice = [slice(None),slice(None),slice(None)]
        selecting_slice[ax1] = slice(0,ax1_idx)
        selecting_slice[ax2] = slice(axis_indices[ax2][w[maxax2idx]],None)
        selecting_slice[axis] = slice(axis_indices[axis][w[maxax2idx]], None) if one_after else slice(None, axis_indices[axis][w[maxax2idx]])
        area_mask[*selecting_slice] = 1
        # extend over max ax1: base case
        ax1_idx = np.max(axis_indices[ax1])
        w = np.where(axis_indices[ax1] == ax1_idx)[0]
        for i in w:
            selecting_slice = [slice(None),slice(None),slice(None)]
            selecting_slice[ax1] = slice(ax1_idx,None)
            selecting_slice[ax2] = axis_indices[ax2][i]
            selecting_slice[axis] = slice(axis_indices[axis][i], None) if one_after else slice(None, axis_indices[axis][i])
            area_mask[*selecting_slice] = 1
        # extend over max ax1: ax2 extensions
        # min ax2
        minax2idx = np.argmin(axis_indices[ax2][w])
        selecting_slice = [slice(None),slice(None),slice(None)]
        selecting_slice[ax1] = slice(ax1_idx,None)
        selecting_slice[ax2] = slice(0,axis_indices[ax2][w[minax2idx]])
        selecting_slice[axis] = slice(axis_indices[axis][w[minax2idx]], None) if one_after else slice(None, axis_indices[axis][w[minax2idx]])
        area_mask[*selecting_slice] = 1
        # max ax2
        maxax2idx = np.argmax(axis_indices[ax2][w])
        selecting_slice = [slice(None),slice(None),slice(None)]
        selecting_slice[ax1] = slice(ax1_idx,None)
        selecting_slice[ax2] = slice(axis_indices[ax2][w[maxax2idx]],None)
        selecting_slice[axis] = slice(axis_indices[axis][w[maxax2idx]], None) if one_after else slice(None, axis_indices[axis][w[maxax2idx]])
        area_mask[*selecting_slice] = 1
    elif slice_by_slice:
        axis_indices, z_indices = _get_extremal_idx_by_z_slice(organ_segmentation, axis, get_largest_index)
        for i in reversed(range(len(z_indices))):
            selecting_slice = slice(axis_indices[i], None) if one_after else slice(None, axis_indices[i])
            if axis == 0:
                indices = (selecting_slice, slice(None), z_indices[i])
            elif axis == 1:
                indices = (slice(None), selecting_slice, z_indices[i])
            area_mask[indices] = 1
        # everything "before" the lowest z index
        selecting_slice = slice(axis_indices[0], None) if one_after else slice(None, axis_indices[0])
        if axis == 0:
            indices = (selecting_slice, slice(None), slice(0,z_indices[0]))
        elif axis == 1:
            indices = (slice(None), selecting_slice, slice(0,z_indices[0]))
        area_mask[indices] = 1
        # everything "after" the highest z index
        selecting_slice = slice(axis_indices[-1], None) if one_after else slice(None, axis_indices[-1])
        if axis == 0:
            indices = (selecting_slice, slice(None), slice(z_indices[-1],None))
        elif axis == 1:
            indices = (slice(None), selecting_slice, slice(z_indices[-1],None))
        area_mask[indices] = 1
    else:
        idx = _get_extremal_idx(organ_segmentation, axis, get_largest_index)
        selecting_slice = slice(idx, None) if one_after else slice(None, idx)
        if axis == 0:
            indices = (selecting_slice, Ellipsis)
        elif axis == 1:
            indices = (slice(None), selecting_slice, slice(None))
        elif axis == 2:
            indices = (Ellipsis, selecting_slice)
        area_mask[indices] = 1
    return area_mask


def borders2args(area_border: str, structure_border: str):
    return _area_border_specs_to_args[area_border], _structure_border_specs_to_args[structure_border]


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
            one_after_ = _area_border_specs_to_args[border]['one_after']
            axis_ = _area_border_specs_to_args[border]['axis']
            for organ_border_specs in specs:
                seg_filename = organ_border_specs['structure'] + '.nii.gz'
                for organ_border_name in organ_border_specs['border']:
                    get_largest_index_ = _area_border_specs_to_args[organ_border_name]['get_largest_index']
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
    def _gen_areas():
        for border in do_first_borders:
            specs = area_specs[border]
            one_after_ = _area_border_specs_to_args[border]['one_after']
            axis_ = _area_border_specs_to_args[border]['axis']
            for organ_border_specs in specs:
                seg_filename = organ_border_specs['structure'] + '.nii.gz'
                for organ_border_name in organ_border_specs['border']:
                    get_largest_index_ = _structure_border_specs_to_args[organ_border_name]['get_largest_index']
                    seg_file_path = os.path.join(
                        path_to_totalseg_segmentations,
                        totalseg_structure_to_task[seg_filename],
                        patient,
                        seg_filename
                    )
                    organ_segmentation = torch.tensor(nib.load(seg_file_path).get_fdata()).to(torch.uint8)
                yield define_area_by_plane(organ_segmentation, axis_, get_largest_index_, one_after_, False, False)  # slice-by-slice and line-by-line always False for z borders
        # do the other borders
        for border, specs in area_specs.items():
            if border in do_first_borders:
                continue
            one_after_ = _area_border_specs_to_args[border]['one_after']
            axis_ = _area_border_specs_to_args[border]['axis']
            for organ_border_specs in specs:
                seg_filename = organ_border_specs['structure'] + '.nii.gz'
                for organ_border_name in organ_border_specs['border']:
                    kwargs = _structure_border_specs_to_args[organ_border_name]
                    seg_file_path = os.path.join(
                        path_to_totalseg_segmentations,
                        totalseg_structure_to_task[seg_filename],
                        patient,
                        seg_filename
                    )
                    organ_segmentation = torch.tensor(nib.load(seg_file_path).get_fdata()).to(torch.uint8)
                yield define_area_by_plane(organ_segmentation, axis_, one_after=one_after_, **kwargs)
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


def get_bbox(label):
    w = np.where(label > 0)
    bbox_coords=[[],[]]
    for ax in range(3):
        lo,hi = w[ax].min(), w[ax].max()
        #print(f'{ax}, {lo}, {hi}')
        bbox_coords[0].append(lo)
        bbox_coords[1].append(hi)
    return bbox_coords


def refine_empty_slices(mask):
    nonzeroslices = sorted(
        torch.where(mask.sum(axis=(0,1)) > 0)[0],
        reverse=True
    )
    for i in range(nonzeroslices.min(), nonzeroslices.max()+1):
        if i not in nonzeroslices:
            mask[...,i] = mask[...,i-1]
    return mask

