
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

from looptrace.ImageHandler import ImageHandler
import dask.array as da
from looptrace import image_io
import tqdm
import numpy as np
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run spot detection on all frames and channels listed in config.')
    parser.add_argument("config_path", help="Config file path")
    parser.add_argument("image_path", help="Path to folder with images to read.")
    args = parser.parse_args()
    try:
        array_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    except KeyError:
        array_id = None
        
    H = ImageHandler(config_path=args.config_path, image_path=args.image_path, pos_id = array_id)


    for i, pos in enumerate(H.image_lists['seq_images_raw_decon']):
        out = da.concatenate([H.images['seq_images_raw_decon'][i], H.images['seq_images_raw2_decon'][i]], axis=0)
        
        z = image_io.create_zarr_store(path=H.image_save_path+os.sep+'seq_images_raw_decon_combined',
                                    name = 'seq_images_raw_decon_combined', 
                                    pos_name = pos,
                                    shape = out.shape, 
                                    dtype = np.uint16,  
                                    chunks = (1,1,1,out.shape[-2], out.shape[-1]))

        n_t = out.shape[0]

        for t in tqdm.tqdm(range(n_t)):
            z[t] = out[t]
