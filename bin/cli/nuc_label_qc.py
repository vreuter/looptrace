
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

from looptrace.ImageHandler import ImageHandler
from looptrace.NucDetector import NucDetector
from looptrace import image_io
import napari
import numpy as np
import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract experimental PSF from bead images.')
    parser.add_argument("config_path", help="Config file path")
    parser.add_argument("image_path", help="Path to folder with images to read.")
    parser.add_argument("--image_save_path", help="(Optional): Path to folder to save images to.", default=None)
    parser.add_argument("--qc", help="(Optional): Additionally run QC (allows edits).", action='store_true')
    parser.add_argument('--raw', help='Use raw images instead of extracted images for visualizing nuclei', action='store_true')
    args = parser.parse_args()
    H = ImageHandler(config_path=args.config_path, image_path=args.image_path, image_save_path=args.image_save_path)
    N = NucDetector(H)
    if 'nuc_images' not in H.images or 'nuc_masks' not in H.images:
        print('Nuclei need to be segmented first.')
        sys.exit()
    
    nuc_imgs = H.images['nuc_images']
    for i, pos in enumerate(H.image_lists['nuc_images']):
        print('Viewing nucleus images from position ', pos)
        if args.raw:
            viewer = napari.view_image(H.images[H.config['nuc_input_name']][i][0], channel_axis=0)
        else:
            viewer = napari.view_image(H.images['nuc_images'][i][0,0])
        if 'nuc_masks' in H.images:
            nuc_mask = H.images['nuc_masks'][i][0,0]
            if args.qc:
                nuc_mask = np.array(nuc_mask)
            masks_layer = viewer.add_labels(nuc_mask)
            masks_layer.n_edit_dimensions = 3
        if 'nuc_classes' in H.images:
            nuc_class = H.images['nuc_classes'][i][0,0]
            if args.qc:
                nuc_class = np.array(nuc_class)
            classes_layer = viewer.add_labels(nuc_class)
            classes_layer.n_edit_dimensions = 3
        napari.run()

        user_input = input('Press enter to continue to next position, or q to quit.')
        if user_input == 'q':
            break

        if args.qc:
            if 'nuc_masks' in H.images:
                image_io.single_position_to_zarr(masks_layer.data.astype(np.uint16), path = N.nuc_masks_path, name = 'nuc_mask', pos_name=pos, axes=('z','y','x'), dtype = np.uint16, chunk_split=(1,1))
                #N.update_masks_after_qc(masks_layer.data.astype(np.uint16), np.array(H.images['nuc_masks'][i]), 'nuc_masks', H.image_lists['nuc_masks'][i])
            if 'nuc_classes' in H.images:  
                image_io.single_position_to_zarr(classes_layer.data.astype(np.uint16), path = N.nuc_classes_path, name = 'nuc_class', pos_name=pos, axes=('z','y','x'), dtype = np.uint16, chunk_split=(1,1))
                #N.update_masks_after_qc(classes_layer.data.astype(np.uint16), np.array(H.images['nuc_classes'][i]), 'nuc_classes', H.image_lists['nuc_classes'][i])
            if 'nuc_masks' in H.images:
                del nuc_mask
                del masks_layer
            if 'nuc_classes' in H.images:
                del nuc_class
                del classes_layer