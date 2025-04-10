from library import define_area_by_specs_with_heuristics, totalseg_tasks, totalseg_tasks_local, get_bbox, refine_empty_slices
from lymph_node_levels_specs import level_specs
import os
from pathlib import Path
import logging
import multiprocessing
import numpy as np
import nibabel as nib
import pandas as pd
import torch


##### CONFIGURATION AND PATHS
## Autopet control
#path_to_ct = '/mnt/nas/autoPET/data/FDG-PET-CT-Lesions_nifti_neg'
#path_to_totalseg_segmentations = '/mnt/nas/database_CAL_immuno_lucie/pet_metrics_project/runs/autopet_control_patients/seg_and_metrics_run_2024_11_22/output/SEG'
#totalseg_structure_to_task = {struct: task for task in totalseg_tasks for struct in os.listdir(os.path.join(path_to_totalseg_segmentations, task, 'PETCT_1bdefef7d5_20060114'))}
#output_dir = '/home/francescocremonesiext/new-areas/aires-ganglionnaires/output/autopet_control_patients'
#path_to_lesion_seg = None

## CAL cancer
path_to_totalseg_segmentations = '/mnt/nas/database_CAL_immuno_lucie/pet_metrics_project/runs/cal_cancer_patients/seg_and_metrics_run_2024_04_09/output/SEG'
path_to_ct = '/mnt/lvssd/common/AI4PET/data_v1.0.0/data/02_intermediate/CAL/CT'
totalseg_structure_to_task = {struct: task for task in totalseg_tasks for struct in os.listdir(os.path.join(path_to_totalseg_segmentations, task, '0000002_20171205'))}
output_dir = '/home/francescocremonesiext/new-areas/aires-ganglionnaires/output/cal_cancer_patients'

# Bouding boxes only?
only_bounding_boxes = False
if only_bounding_boxes:
    output_dir += '_bb'
    
# Figure out currently checked out git commit
with open('./.git/HEAD', 'r') as f:
    lines = f.readlines()
ref = lines[0].split(':')[1].strip()
with open(f'./.git/{ref}', 'r') as f:
    lines = f.readlines()
gitsha = lines[0][:10]
output_dir += f'_{gitsha}'
##### END CONFIGURATION AND PATHS

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = multiprocessing.get_logger() 
rootLogger.setLevel(logging.DEBUG)

outlogfile = f"{output_dir}/aires-ganglionnaires.log"
if not os.path.exists(outlogfile):
    Path(outlogfile).touch()
fileHandler = logging.FileHandler(outlogfile)
fileHandler.setFormatter(logFormatter)
fileHandler.setLevel(logging.DEBUG)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
consoleHandler.setLevel(logging.INFO)
rootLogger.addHandler(consoleHandler)


def process_one_patient(patient):
    volumes = list()
    bboxes = list()
    result_levels = list()

    rootLogger.info(f'Patient {patient}')
    
    # Generic admin such as creating directories
    out_dir = f'{output_dir}/SEG/{patient}'
    if os.path.exists(out_dir):
        rootLogger.info(f'Output directory already exists for {patient}. Skipping processing.')
        return
    os.makedirs(out_dir, exist_ok = True)

    # Load CT data
    if False:  # for Autopet
        patient_nodate = '_'.join(patient.split('_')[:2])
        imgdir = os.listdir(os.path.join(path_to_ct, patient_nodate))[0]
        ct_img = nib.load(os.path.join(path_to_ct, patient_nodate, imgdir, 'CT.nii.gz'))
        rootLogger.debug(f'patient no date {patient_nodate}')
    else:
        try:
            ct_img = nib.load(os.path.join(path_to_ct, f'{patient}.nii.gz'))
        except FileNotFoundError:
            rootLogger.error(f'Could not find {path_to_ct}/{patient}.nii.gz Skipping.')
            return
        rootLogger.debug(f'patient {patient}')

    with torch.no_grad():
        # Generate each area one-by-one in a loop
        combined = None
        for level, specs in level_specs.items():
            rootLogger.info(f'Computing {level} for {patient}')
            try:
                # 1. Start by creating a "box" that defines the area
                level_mask = define_area_by_specs_with_heuristics(specs, 
                        patient, 
                        path_to_totalseg_segmentations, 
                        totalseg_structure_to_task)
                bbox = get_bbox(level_mask)
                # 2. Refinement: remove all other totalsegmentator structures
                if not only_bounding_boxes:
                    if combined is None:  # Only for the first time: we combine all other totalsegmentator masks 
                        rootLogger.info(f'Combining all totalseg masks for {patient}')
                        combined = torch.zeros_like(level_mask, dtype=torch.uint8)
                        for structure, task in totalseg_structure_to_task.items():
                            if '_fat.nii' in structure:
                                logging.debug(f'Skipping {structure}')
                                continue
                            segdata = torch.tensor(nib.load(os.path.join(
                                                path_to_totalseg_segmentations,
                                                task,
                                                patient,
                                                structure
                                            )).get_fdata()).to(torch.uint8)
                            combined = torch.clip(combined + segdata, 0, 1)
                        rootLogger.debug(f'Finished combining all masks for {patient}')
                    level_mask = level_mask*(1 - combined)
                # 3. Refinement: remove space outside of the body
                rootLogger.info(f'Removing space outside body for {patient} {level}')
                body = torch.tensor(nib.load(os.path.join(
                                    path_to_totalseg_segmentations,
                                    'body',
                                    patient,
                                    'body.nii.gz'
                                )).get_fdata()).to(torch.uint8)
                level_mask = level_mask*body
                # 4. Refinement: fill empty slices
                rootLogger.info(f'Refining empty slices for {patient} {level}')
                level_mask = refine_empty_slices(level_mask)
                # Save results
                result_levels.append(level)
                volumes.append(level_mask.sum())
                bboxes.append(bbox)
                rootLogger.debug(f'Saving {level}.nii.gz for {patient}')
                level_mask = level_mask.cpu().numpy().astype(np.uint8)
                ni_img = nib.Nifti1Image(level_mask, ct_img.affine, ct_img.header)
                ni_img.set_data_dtype(np.uint8)
                nib.save(ni_img, f'{output_dir}/SEG/{patient}/{level}.nii.gz')   
            except Exception as e:
                rootLogger.error(f'Could not process {level} for {patient} because of {e}')
    rootLogger.debug(f'Saving volumes for {patient}')
    pixdim = ct_img.header['pixdim']
    vox_vol = pixdim[1] * pixdim[2] * pixdim[3]
    df = pd.DataFrame({'patient': [patient]*len(volumes), 'level': result_levels, 'volume [vx]': volumes, 'voxel_volume [mm^3/vx]': [vox_vol]*len(volumes)})
    for ax in range(3):
        df[f'bbox_ax{ax}_lo'] = [box[0][ax] for box in bboxes]
        df[f'bbox_ax{ax}_hi'] = [box[1][ax] for box in bboxes]
    os.makedirs(f'{output_dir}/VOL/{patient}', exist_ok=True)
    df.to_csv(f'{output_dir}/VOL/{patient}/volumes.csv', index=False)


if __name__ == '__main__':
    #patients = os.listdir(os.path.join(path_to_totalseg_segmentations, 'total'))
    patients_df = pd.read_csv('compliant_scan_ids_cancer_patients.csv')
    patients = patients_df['compliant_scan_ids'].values.tolist()
    rootLogger.info('Starting now')
    pool = multiprocessing.Pool(processes=8)
    pool.map(process_one_patient, patients)





