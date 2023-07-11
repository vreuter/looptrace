
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import dask.array as da
from looptrace import image_io
import tqdm
import numpy as np
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert datasets to OME-ZARR (for now ND2, will make general as needed).')
    parser.add_argument("--input_folders", nargs='+', help='<Required> Folderpath(s) with ND2 images to convert to zarr. Assumes Time00000_Point0000_ naming convention.', required=True)
    parser.add_argument("--output_folder", help='<Required> Folderpath to save zarr images.', required=True)
    parser.add_argument("--n_pos", help='Number of positions to convert (if not using array jobs)')
    args = parser.parse_args()
    print(args)
    try:
        array_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    except KeyError:
        array_id = None

    if array_id is None:
        for pos_id in tqdm.tqdm(range(int(args.n_pos))):
            imgs = []
            for f in args.input_folders:
                folder_imgs, folder_positions, folder_metadata = image_io.stack_nd2_to_dask(f, position_id=pos_id)
                imgs.append(folder_imgs[0])
            imgs = da.concatenate(imgs, axis=0)
            print(folder_metadata)
            z = image_io.create_zarr_store(path=args.output_folder,
                                    name = os.path.basename(args.output_folder), 
                                    pos_name = 'P'+str(pos_id+1).zfill(4)+'.zarr',
                                    shape = imgs.shape, 
                                    dtype = np.uint16,  
                                    chunks = (1,1,1,imgs.shape[-2], imgs.shape[-1]),
                                    metadata = folder_metadata,
                                    voxel_size = folder_metadata['voxel_size'])
        
            n_t = imgs.shape[0]
            for t in tqdm.tqdm(range(n_t)):
                z[t] = imgs[t]

    else:
        imgs = []
        for f in args.input_folders:
            folder_imgs, folder_positions, folder_metadata = image_io.stack_nd2_to_dask(f, position_id=array_id)
            imgs.append(folder_imgs[0])
        imgs = da.concatenate(imgs, axis=0)
        z = image_io.create_zarr_store(path=args.output_folder,
                                name = os.path.basename(args.output_folder), 
                                pos_name = 'P'+str(array_id+1).zfill(4)+'.zarr',
                                shape = imgs.shape, 
                                dtype = np.uint16,  
                                chunks = (1,1,1,imgs.shape[-2], imgs.shape[-1]), 
                                metadata = folder_metadata,
                                voxel_size = folder_metadata['voxel_size'])
        n_t = imgs.shape[0]
        for t in tqdm.tqdm(range(n_t)):
            z[t] = imgs[t]