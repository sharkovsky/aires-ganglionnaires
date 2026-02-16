totalseg_tasks = ['total', 'tissue', 'head_muscles', 'head_glands_cavities', 'headneck_bones_vessels', 'headneck_muscles']
totalseg_tasks_local = ['total', 'tissue_4_types', 'head_muscles', 'head_glands_cavities', 'headneck_bones_vessels', 'headneck_muscles']

levelname2num_rpa = {
#     "background": 0,
     "level_ia_left": [1],
     "level_ia_right": [2],
     "level_ib_left": [3],
     "level_ib_right": [4],
     "level_ii_left": [5],
     "level_ii_right": [6],
     "level_iii_left": [7],
     "level_iii_right": [8],
     "level_iv_left": [9],
     "level_iv_right": [10],
     "level_v_left": [11],
     "level_v_right": [12],
     "level_rp_left": [13],
     "level_rp_right": [14]
 }

levelname2num_cal = {
#     "background": 0,
     "level_ia": [1,2],
     "level_ib_left": [3],
     "level_ib_right": [4],
     "level_ii_left": [5],
     "level_ii_right": [6],
     "level_iii_left": [7],
     "level_iii_right": [8],
     "level_iv_left": [9],
     "level_iv_right": [10],
     "level_v_left": [11],
     "level_v_right": [12],
     "level_rp_left": [13],
     "level_rp_right": [14]
 }
levelname2num_cal['level_iia_left'] = [5]
levelname2num_cal['level_iib_left'] = [5]
levelname2num_cal['level_iia_right'] = [6]
levelname2num_cal['level_iib_right'] = [6]

levelname2num_lnctv = {
     "level_ib_left": [6],
     "level_ib_right": [5],
     "level_ii_left": [2],
     "level_ii_right": [1],
     "level_iii_left": [2],
     "level_iii_right": [1],
     "level_iv_left": [4],
     "level_iv_right": [3],
     "level_va_left": [2],
     "level_va_right": [1],
     "level_vb_left": [4],
     "level_vb_right": [3]
}

levelnames_combined = {
    'level_ib':  ['ib'],
    'level_others': ['ii', 'iii', 'iv', 'v', 'va', 'vb', 'iia', 'iib']
}

