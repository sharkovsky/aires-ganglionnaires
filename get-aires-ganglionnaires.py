from library import define_area_by_specs, totalseg_tasks
from lymph-node-levels-specs import level_specs
import os
import logging
import multiprocessing
import numpy as np
import nibabel as nib
import pandas as pd

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = multiprocessing.get_logger() 
rootLogger.setLevel(logging.DEBUG)

fileHandler = logging.FileHandler("./logs/aires-ganglionnaires.log")
fileHandler.setFormatter(logFormatter)
fileHandler.setLevel(logging.DEBUG)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
consoleHandler.setLevel(logging.INFO)
rootLogger.addHandler(consoleHandler)


path_to_totalseg_segmentations = '/mnt/nas/database_CAL_immuno_lucie/pet_metrics_project/runs/autopet_control_patients/seg_and_metrics_run_2024_11_22/output/SEG'
path_to_ct = '/mnt/nas/autoPET/data/FDG-PET-CT-Lesions_nifti_neg'

totalseg_structure_to_task = {struct: task for task in totalseg_tasks for struct in os.listdir(os.path.join(path_to_totalseg_segmentations, task, 'PETCT_1bdefef7d5_20060114'))}


def process_one_patient(patient):
    volumes = list()
    result_levels = list()

    rootLogger.info(f'Patient {patient}')
    
    out_dir = f'/home/francescocremonesiext/new-areas/aires-ganglionnaires/output/autopet_control_patients/{patient}'
    if os.path.exists(out_dir):
        rootLogger.debug(f'Output directory already exists for {patient}. Skipping processing.')
        return
    os.makedirs(out_dir, exist_ok = True)
    patient_nodate = '_'.join(patient.split('_')[:2])
    imgdir = os.listdir(os.path.join(path_to_ct, patient_nodate))[0]
    ct_img = nib.load(os.path.join(path_to_ct, patient_nodate, imgdir, 'CT.nii.gz'))

    rootLogger.debug(f'patient no date {patient_nodate}')

    combined = None

    for level, specs in level_specs.items():
        try:
            rootLogger.info(f'Computing {level} for {patient}')
            level_mask = define_area_by_specs(specs, patient, path_to_totalseg_segmentations, totalseg_structure_to_task)
            if combined is None:
                rootLogger.info(f'Combining all totalseg masks for {patient}')
                combined = np.zeros_like(level_mask, dtype=np.int32)
                for structure, task in totalseg_structure_to_task.items():
                    if 'subcutaneous_fat' in structure:
                        logging.debug(f'Skipping {structure}')
                        continue
                    segdata = nib.load(os.path.join(
                                        path_to_totalseg_segmentations,
                                        task,
                                        patient,
                                        structure
                                    )).get_fdata()
                    combined = np.clip(combined + segdata.astype(np.int32), 0, 1)
                    rootLogger.debug(f'Finished combining all masks for {patient}')
            level_mask *= 1 - combined
            result_levels.append(level)
            volumes.append(level_mask.sum())
            rootLogger.debug(f'Saving {level}.nii.gz for {patient}')
            ni_img = nib.Nifti1Image(level_mask, ct_img.affine, ct_img.header)
            nib.save(ni_img, f'/home/francescocremonesiext/new-areas/aires-ganglionnaires/output/autopet_control_patients/{patient}/{level}.nii.gz')   
            rootLogger.debug(f'Saving volumes for {patient}')
        except Exception as e:
            rootLogger.error(f'Could not process {level} for {patient}')
    pixdim = ct_img.header['pixdim']
    vox_vol = pixdim[1] * pixdim[2] * pixdim[3]
    df = pd.DataFrame({'patient': [patient]*len(volumes), 'level': result_levels, 'volume [vx]': volumes, 'voxel_volume [mm^3/vx]': [vox_vol]*len(volumes)})
    df.to_csv(f'/home/francescocremonesiext/new-areas/aires-ganglionnaires/output/autopet_control_patients/{patient}/volumes.csv', index=False)


if __name__ == '__main__':
    patients = os.listdir(os.path.join(path_to_totalseg_segmentations, 'total'))
    rootLogger.info('Starting now')
    pool = multiprocessing.Pool(processes=4)
    pool.map(process_one_patient, patients)





