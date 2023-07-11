"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

from looptrace.ImageHandler import ImageHandler
from looptrace import image_processing_functions as ip
import os
import argparse
import pandas as pd
import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Assign detected spot ROIs to nucleus labels/classes.')
    parser.add_argument("--config_path", help="Config file path")
    parser.add_argument("--image_path", help="Path to folder with images to read.")
    args = parser.parse_args()
    try:
        array_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    except KeyError:
        array_id = None
    H = ImageHandler(config_path=args.config_path, image_path=args.image_path, pos_id = array_id)
    all_rois = []
    print('Assigning spots to nuclei labels.')
    for i, pos in tqdm.tqdm(enumerate(H.image_lists[H.config['spot_input_name']])):
        try:
            nuc_drifts = H.tables[H.config['nuc_input_name']+'_drift_correction'].query('position == @pos')
            if i == 0:
                print('Using nucleus drift correction.')
        except KeyError:
            nuc_drifts = None
        try:
            spot_drifts = H.tables[H.config['spot_input_name']+'_drift_correction'].query('position == @pos')
            if i==0:
                print('Using spot drift correction.')
        except KeyError:
            spot_drifts = None

        rois = H.tables[H.config['spot_input_name']+'_rois'].query('position == @pos')
        if len(rois) == 0:
            continue

        if 'nuc_masks' in H.images:
            rois = ip.filter_rois_in_nucs(rois, H.images['nuc_masks'][i][0,0], new_col='nuc_label', nuc_drifts=nuc_drifts, nuc_target_frame=H.config['nuc_ref_frame'], spot_drifts = spot_drifts)
        if 'nuc_classes' in H.images:
            rois = ip.filter_rois_in_nucs(rois, H.images['nuc_classes'][i][0,0], new_col='nuc_class', nuc_drifts=nuc_drifts, nuc_target_frame=H.config['nuc_ref_frame'], spot_drifts = spot_drifts)
        all_rois.append(rois.copy())

    all_rois = pd.concat(all_rois).sort_values(['position', 'frame'])
    if array_id is not None:
        all_rois.to_csv(H.out_path+H.config['spot_input_name']+'_rois_'+str(array_id).zfill(4)+'.csv')
    else:
        all_rois.to_csv(H.out_path+H.config['spot_input_name']+'_rois.csv')
        if H.config['spot_input_name']+'_traces' in H.tables:
            print('Assigning ids to traces.')
            traces = H.tables[H.config['spot_input_name']+'_traces'].copy()
            if 'nuc_masks' in H.images:
                traces.loc[:,'nuc_label'] = traces['trace_id'].map(all_rois['nuc_label'].to_dict())
            if 'nuc_classes' in H.images:
                traces.loc[:,'nuc_class'] = traces['trace_id'].map(all_rois['nuc_class'].to_dict())
            traces.sort_values(['trace_id','frame']).to_csv(H.out_path+H.config['spot_input_name']+'_traces.csv')