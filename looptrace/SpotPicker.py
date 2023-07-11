# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

from scipy import ndimage as ndi
from looptrace import image_processing_functions as ip
from looptrace import image_io
import pandas as pd
import numpy as np
import random
from skimage.measure import regionprops_table
import tqdm
import os

class SpotPicker:
    def __init__(self, image_handler, roi_name = None):
        self.image_handler = image_handler
        self.config = image_handler.config
        self.images = self.image_handler.images[self.config['spot_input_name']]
        self.pos_list = self.image_handler.image_lists[self.config['spot_input_name']]
        if roi_name == None:
            self.roi_name = self.config['spot_input_name']
        else:
            self.roi_name = roi_name
        self.roi_path = self.image_handler.out_path+self.roi_name+'_rois.csv'
        self.dc_roi_path = self.image_handler.out_path+self.roi_name+'_dc_rois.csv'
        if self.image_handler.pos_id is not None:
            self.roi_path = self.image_handler.out_path+self.roi_name+'_rois.csv'[:-4]+'_'+str(self.image_handler.pos_id).zfill(4)+'.csv'

    def rois_from_spots(self, preview_pos = None):

        # Detects spot ROIS based on a pre-trained random forest model (training is done in separate notebook)
        # 

        from skimage.morphology import remove_small_objects
        from skimage.segmentation import expand_labels
        import tqdm
        
        spot_ch = self.config['spot_ch']
        if not isinstance(spot_ch, list):
            spot_ch = [spot_ch]
        spot_frame = self.config['spot_frame']
        if not isinstance(spot_frame, list):
            spot_frame = [spot_frame]
        spot_ds = self.config['spot_downsample']

        try:
            spot_min_size = self.config['spot_min_size']
        except KeyError:
            spot_min_size = 3

        try:
            spot_dilate = self.config['spot_dilate']
        except KeyError:
            spot_dilate = 1

        try:
            min_dist = self.config['min_spot_dist']
        except KeyError: #Legacy config.
            min_dist = None

        try:
            substract_crosstalk = self.config['subtract_crosstalk']
            crosstalk_ch = self.config['crosstalk_ch']
        except KeyError: #Legacy config.
            substract_crosstalk = False

        try:
            crosstalk_frame = self.config['crosstalk_frame']
        except KeyError:
            crosstalk_frame = -1

        try:
            spot_threshold = self.config['spot_threshold']
        except KeyError:
            spot_threshold = 1000

        if not isinstance(spot_threshold, list):
            spot_threshold = [spot_threshold]*len(spot_frame)
            
        try:
            detect_method = self.config['detection_method']
        except KeyError: #Legacy config.
            detect_method = 'dog'

        rois = []


        for i, pos in tqdm.tqdm(enumerate(self.pos_list)):
            for j, frame in tqdm.tqdm(enumerate(spot_frame)):
                for k, ch in enumerate(spot_ch):
                #print(f'Detecting spots in position {position}, frame {frame}, ch {ch}.',  end=' ')
                    img = self.images[i][frame, ch, ::spot_ds, ::spot_ds, ::spot_ds].compute()

                    if substract_crosstalk:
                        if crosstalk_frame == -1:
                            crosstalk_img = self.images[i][frame, crosstalk_ch, ::spot_ds, ::spot_ds, ::spot_ds].compute()
                        else:
                            crosstalk_img = self.images[i][crosstalk_frame, crosstalk_ch, ::spot_ds, ::spot_ds, ::spot_ds].compute()
                        img, _ = ip.subtract_crosstalk(img, crosstalk_img, threshold=0)

                    if detect_method == 'rf':
                        import pickle
                        from skimage import feature, future
                        from functools import partial
                        sigma_min = 1
                        sigma_max = 16
                        features_func = partial(feature.multiscale_basic_features,
                                                intensity=True, edges=True, texture=False,
                                                sigma_min=sigma_min, sigma_max=sigma_max)
                        clf = pickle.load(open(os.path.dirname(self.image_handler.out_path)+os.sep+'spots_rf_model.pickle', 'rb'))
                        features = features_func(img)
                        mask = future.predict_segmenter(features, clf)
                        mask = mask - 1
                    
                    elif detect_method == 'intensity':
                        mask = img > spot_threshold[j]
                    elif detect_method == 'sbr':
                        mask = img > (spot_threshold[j] * np.median(img))
                    elif detect_method == 'dog':
                        from skimage.filters import gaussian
                        #img = white_tophat(image=input_img, footprint=ball(2))
                        filt_img = gaussian(img, 0.8)-gaussian(img,1.3)
                        filt_img = filt_img/gaussian(filt_img, 3)
                        filt_img = (filt_img-np.mean(filt_img))/np.std(filt_img)
                        mask = filt_img > spot_threshold[j]

                    else:
                        print('Unknown spot detection method, returning.')


                    struct = ndi.generate_binary_structure(img.ndim, 2)
                    labels, n_obj = ndi.label(mask, structure=struct)

                    if n_obj > 1: #Do not need this with area filtering below.
                        labels = remove_small_objects(labels, min_size=spot_min_size)
                        labels = expand_labels(labels, spot_dilate)
                    
                    if preview_pos is not None:
                        import napari
                        n = napari.view_image(img)
                        n.add_labels(labels)
                        input('Press any key to to proceed.')
                        return

                    if np.all(labels == 0): #If there are no labels anymore:
                        return pd.DataFrame(columns=['label', 'z_min','y_min','x_min','z_max','y_max','x_max','area','zc','yc','xc','intensity_mean']), labels
                    else:
                        spot_props = regionprops_table(labels, img, properties=('label', 'bbox', 'area', 'centroid_weighted', 'intensity_mean'))
                        spot_props = pd.DataFrame(spot_props)
                        #spot_props = spot_props.query('area > 10')

                        spot_props = spot_props.rename(columns={'centroid_weighted-0': 'zc',
                                                            'centroid_weighted-1': 'yc',
                                                            'centroid_weighted-2': 'xc',
                                                            'bbox-0': 'z_min',
                                                            'bbox-1': 'y_min',
                                                            'bbox-2': 'x_min',
                                                            'bbox-3': 'z_max',
                                                            'bbox-4': 'y_max',
                                                            'bbox-5': 'x_max'})
                    
                        spot_props = spot_props.reset_index(drop=True)
                        spot_props = spot_props.rename(columns={'index':'roi_id'})
                        spot_props[['z_min', 'y_min', 'x_min', 'z_max', 'y_max', 'x_max', 'zc', 'yc', 'xc']] = spot_props[['z_min', 'y_min', 'x_min', 'z_max', 'y_max', 'x_max', 'zc', 'yc', 'xc']]*spot_ds
                    
                        spot_props['position'] = pos
                        spot_props['frame'] = frame
                        spot_props['ch'] = ch
                        print(f'Found {str(len(spot_props))} spots in position {pos}, frame {str(frame)}, channel {str(ch)}.')
                        rois.append(spot_props)

        rois = pd.concat(rois)
        rois = rois.sort_values(['position', 'frame'])
        rois.to_csv(self.roi_path)
        self.image_handler.load_tables()

        print(f'Filtering complete, {len(rois)} ROIs after filtering.')


    def rois_from_beads(self):

        # Detect bead ROIS based on drift correction parameters in config.

        print('Detecting bead ROIs for tracing.')
        all_rois = []
        n_fields = self.config['bead_trace_fields']
        n_beads = self.config['bead_trace_number']
        for pos in tqdm.tqdm(sorted(random.sample(self.pos_list, k=n_fields))):
            pos_index = self.image_handler.image_lists[self.config['spot_input_name']].index(pos)
            ref_frame = self.config['reg_ref_frame']
            ref_ch = self.config['reg_ch_template']
            threshold = self.config['bead_threshold']
            min_bead_int = self.config['min_bead_intensity']

            t_img = self.images[pos_index][ref_frame, ref_ch].compute()
            t_img_label,num_labels = ndi.label(t_img>threshold)
            #t_img_maxima=np.array(ndi.measurements.maximum_position(t_img, 
            #                                            labels=t_img_label, 
            #                                            index=np.random.choice(np.arange(1,num_labels), size=n_points*2)))
            
            spot_props = pd.DataFrame(regionprops_table(t_img_label, t_img, properties=('label', 'centroid', 'max_intensity')))
            spot_props = spot_props.query('max_intensity > @min_bead_int').sample(n=n_beads, random_state=1)
            
            spot_props.drop(['label'], axis=1, inplace=True)
            spot_props.rename(columns={'centroid-0': 'zc',
                                        'centroid-1': 'yc',
                                        'centroid-2': 'xc',
                                        'index':'roi_id_pos'},
                                        inplace = True)

            spot_props['position'] = pos
            spot_props['frame'] = ref_frame
            spot_props['ch'] = ref_ch
            print('Detected beads in position', pos, spot_props)
            all_rois.append(spot_props)
        
        output = pd.concat(all_rois)
        output=output.reset_index().rename(columns={'index':'roi_id'}).sort_values(['position', 'frame', 'roi_id'])
        rois = ip.roi_center_to_bbox(output, roi_size = self.config['bead_roi_size'])

        self.image_handler.bead_rois = rois
        rois.to_csv(self.roi_path+'_beads.csv')
        self.image_handler.load_tables()
        return rois
    
    def remove_nearby_rois(self):
        print('Filtering rois to remove proximal rois.')
        try:
            preserve = self.config['preserve_spot']
        except KeyError:
            preserve = 'brightest'

        all_rois = []
        roi_table = self.image_handler.tables[self.roi_name+'_rois']
        for position, pos_rois in roi_table.copy().groupby('position'):
            if self.config['spot_input_name']+'_drift_correction' in self.image_handler.tables:
                frames = sorted(list(pos_rois['frame'].unique()))
                for i, frame in enumerate(frames):
                    sel_dc = self.image_handler.tables[self.config['spot_input_name']+'_drift_correction'].query('position == @position & frame == @frame')[['z_px_course', 'y_px_course', 'x_px_course']].to_numpy()
                    #print(sel_dc, pos_rois)
                    pos_rois.loc[pos_rois['frame'] == frame, ['zc', 'yc', 'xc']] = pos_rois.loc[pos_rois['frame'] == frame, ['zc', 'yc', 'xc']] - sel_dc
                    #print(sel_dc, pos_rois)
                print('Drift correction applied, resulting rois: ', pos_rois)
            else:
                print('No drift correction applied, continuing.')
                pass

            if preserve == 'brightest':
                weights = pos_rois['intensity_mean'].to_numpy()
            else:
                weights = None

            remove_idx = ip.get_indexes_of_nearby_rois(pos_rois[['zc', 'yc', 'xc']].to_numpy(), min_dist = self.config['min_spot_dist'], weights = weights)
            #print('Removing overlapping ROIs at following indexes: ', position, remove_idx, remove_idx.shape, pos_rois)
            new_rois = roi_table[roi_table['position']==position].reset_index(drop=True).drop(remove_idx).reset_index(drop=True) #Reset the index to fit with the index from the function above.
            all_rois.append(new_rois)
        all_rois = pd.concat(all_rois).reset_index(drop=True)
        all_rois.to_csv(self.roi_path)
        self.image_handler.load_tables()

    def make_dc_rois_all_frames(self):
        #Precalculate all ROIs for extracting spot images, based on identified ROIs and precalculated drifts between time frames.
        print('Generating list of all ROIs for tracing:')
        try:
            spot_extract_ch = self.config['spot_extract_ch']
        except KeyError:
            spot_extract_ch = self.config['spot_ch']
        if not isinstance(spot_extract_ch, list):
            spot_extract_ch = [spot_extract_ch]
        #positions = sorted(list(self.roi_table.position.unique()))
        all_rois = []
        for i, roi in tqdm.tqdm(self.image_handler.tables[self.roi_name+'_rois'].iterrows(), total=len(self.image_handler.tables[self.roi_name+'_rois'])):
            pos = roi['position']
            pos_index = self.image_handler.image_lists[self.config['spot_input_name']].index(pos)#positions.index(pos)
            dc_pos_name = self.image_handler.image_lists[self.config['reg_input_moving']][pos_index]
            sel_dc = self.image_handler.tables[self.config['spot_input_name']+'_drift_correction'].query('position == @dc_pos_name')
            ref_frame = roi['frame']

            ref_offset = sel_dc.query('frame == @ref_frame')

            for ch in spot_extract_ch:
                Z, Y, X = self.images[pos_index][0,ch].shape[-3:]
                for j, dc_frame in sel_dc.iterrows():
                    z_drift_course = int(dc_frame['z_px_course']) - int(ref_offset['z_px_course'])
                    y_drift_course = int(dc_frame['y_px_course']) - int(ref_offset['y_px_course'])
                    x_drift_course = int(dc_frame['x_px_course']) - int(ref_offset['x_px_course'])

                    z_min = max(roi['z_min'] - z_drift_course, 0)
                    z_max = min(roi['z_max'] - z_drift_course, Z)
                    y_min = max(roi['y_min'] - y_drift_course, 0)
                    y_max = min(roi['y_max'] - y_drift_course, Y)
                    x_min = max(roi['x_min'] - x_drift_course, 0)
                    x_max = min(roi['x_max'] - x_drift_course, X)

                    pad_z_min = abs(min(0,z_min))
                    pad_z_max = abs(max(0,z_max-Z))
                    pad_y_min = abs(min(0,y_min))
                    pad_y_max = abs(max(0,y_max-Y))
                    pad_x_min = abs(min(0,x_min))
                    pad_x_max = abs(max(0,x_max-X))

                    #print('Appending ', s)
                    all_rois.append([pos, pos_index, roi.name, dc_frame['frame'], ref_frame, roi.ch, ch, 
                                    z_min, z_max, y_min, y_max, x_min, x_max, 
                                    pad_z_min, pad_z_max, pad_y_min, pad_y_max, pad_x_min, pad_x_max,
                                    z_drift_course, y_drift_course, x_drift_course, 
                                    dc_frame['z_px_fine'], dc_frame['y_px_fine'], dc_frame['x_px_fine']])

        self.all_rois = pd.DataFrame(all_rois, columns=['position', 'pos_index', 'roi_id', 'frame', 'ref_frame', 'ref_ch', 'trace_ch', 
                                'z_min', 'z_max', 'y_min', 'y_max', 'x_min', 'x_max',
                                'pad_z_min', 'pad_z_max', 'pad_y_min', 'pad_y_max', 'pad_x_min', 'pad_x_max', 
                                'z_px_course', 'y_px_course', 'x_px_course',
                                'z_px_fine', 'y_px_fine', 'x_px_fine'])
        self.all_rois = self.all_rois.sort_values(['roi_id','frame']).reset_index(drop=True)
        print(self.all_rois)
        self.all_rois.to_csv(self.dc_roi_path)
        self.image_handler.load_tables()

    # def extract_single_roi_img(self, single_roi):
    #     #Function to extract single ROI lazily without loading entire stack in RAM.
    #     #Depending on chunking of original data can be more or less performant.

    #     roi_image_size = tuple(self.config['roi_image_size'])
    #     p = single_roi['pos_index']
    #     t = single_roi['frame']
    #     c = single_roi['ch']
    #     z = slice(single_roi['z_min'], single_roi['z_max'])
    #     y = slice(single_roi['y_min'], single_roi['y_max'])
    #     x = slice(single_roi['x_min'], single_roi['x_max'])
    #     pad = ( (single_roi['pad_z_min'], single_roi['pad_z_max']),
    #             (single_roi['pad_y_min'], single_roi['pad_y_max']),
    #             (single_roi['pad_x_min'], single_roi['pad_x_max']))

    #     try:
    #         roi_img = np.array(self.images[p][t, c, z, y, x])

    #         #If microscope drifted, ROI could be outside image. Correct for this:
    #         if pad != ((0,0),(0,0),(0,0)):
    #             #print('Padding ', pad)
    #             roi_img = np.pad(roi_img, pad, mode='edge')

    #     except ValueError: # ROI collection failed for some reason
    #         roi_img = np.zeros(roi_image_size, dtype=np.float32)

    #     #print(p, t, c, z, y, x)
    #     return roi_img  #{'p':p, 't':t, 'c':c, 'z':z, 'y':y, 'x':x, 'img':roi_img}

    def extract_single_roi_img_inmem(self, single_roi, images):
        # Function for extracting a single cropped region defined by ROI from a larger 3D image.
        z = slice(single_roi['z_min'], single_roi['z_max'])
        y = slice(single_roi['y_min'], single_roi['y_max'])
        x = slice(single_roi['x_min'], single_roi['x_max'])
        pad = ( (single_roi['pad_z_min'], single_roi['pad_z_max']),
                (single_roi['pad_y_min'], single_roi['pad_y_max']),
                (single_roi['pad_x_min'], single_roi['pad_x_max']))

        try:
            roi_img = np.array(images[z, y, x])

            #If microscope drifted, ROI could be outside image. Correct for this:
            if pad != ((0,0),(0,0),(0,0)):
                print('Padding.')
                roi_img = np.pad(roi_img, pad, mode='edge')

        except ValueError: # ROI collection failed for some reason
            roi_img = np.zeros((np.abs(z.stop-z.start), np.abs(y.stop-y.start), np.abs(x.stop-x.start)), dtype=np.float32)

        return roi_img  

    def gen_roi_imgs_inmem(self):
        from numpy.lib.format import open_memmap
        try:
            spot_extract_name = self.config['spot_extract_name']
        except KeyError:
            spot_extract_name = self.config['spot_input_name']
        try:
            spot_extract_ch = self.config['spot_extract_ch']
        except KeyError:
            spot_extract_ch = self.config['spot_ch']
        if not isinstance(spot_extract_ch, list):
            spot_extract_ch = [spot_extract_ch]

        # Load full stacks into memory to extract spots.
        # Not the most elegant, but depending on the chunking of the original data it is often more performant than loading subsegments.
        rois = self.image_handler.tables[self.roi_name+'_dc_rois']

        if not os.path.isdir(self.image_handler.image_save_path+os.sep+'spot_images_dir'):
            os.mkdir(self.image_handler.image_save_path+os.sep+'spot_images_dir')

        for pos, pos_group in tqdm.tqdm(rois[rois.position.isin(self.pos_list)].groupby('position')):
            pos_index = self.image_handler.image_lists[self.config['spot_input_name']].index(pos)
            #print(full_image.shape)
            f_id = 0
            n_frames = len(pos_group.frame.unique())
            n_ch = len(pos_group.trace_ch.unique())
            #print(n_frames)
            for frame, frame_group in tqdm.tqdm(pos_group.groupby('frame')):
                ch_id = 0
                for ch, ch_group in tqdm.tqdm(frame_group.groupby('trace_ch')):
                    image_stack = np.array(self.image_handler.images[spot_extract_name][pos_index][int(frame), int(ch)])
                    for i, roi in ch_group.iterrows():
                        roi_img = self.extract_single_roi_img_inmem(roi, image_stack).astype(np.uint16)
                        fn = self.image_handler.image_save_path+os.sep+'spot_images_dir'+os.sep+str(pos)+'_'+str(roi['roi_id']).zfill(5)+'.npy'
                        if f_id == 0:
                            arr = open_memmap(fn, mode='w+', dtype = roi_img.dtype, shape=(n_frames,n_ch)+roi_img.shape)
                            arr[f_id, ch_id] = roi_img
                            arr.flush()
                        else:
                            arr = open_memmap(fn, mode='r+')
                            try:
                                arr[f_id, ch_id] = roi_img
                                arr.flush()
                                #arr[f_id] = np.append(arr[f_id], np.expand_dims(roi_img,0).copy(), axis=0)
                            except ValueError: #Edge case: ROI fetching has failed giving strange shaped ROI, just leave the zeros as is.
                                pass
                                # roi_stack = np.append(roi_stack, np.expand_dims(np.zeros_like(roi_stack[0]), 0), axis=0)
                        #np.save(self.config['image_path']+os.sep+'spot_images_dir'+os.sep+rn+'.npy', roi_stack)
                        #print(roi_array)
                        #roi_array_padded.append(ip.pad_to_shape(roi, shape = roi_image_size, mode = 'minimum'))
                    ch_id += 1
                f_id += 1
        #if self.image_handler.pos_id is None: #Only do this if not running distributed, otherwise need to collect and zip later.
        #    image_io.zip_folder(self.image_handler.image_save_path+os.sep+'spot_images_dir', 'spot_images.npz', remove_folder = True)
        #    self.image_handler.images['spot_images'] = image_io.NPZ_wrapper(self.image_handler.image_save_path+os.sep+'spot_images.npz')

            
            #for j, pos_roi in enumerate(pos_rois):
            #    roi_array[str(pos)+'_'+str(j).zfill(5)] = pos_roi.copy()
        
        #print(roi_array.keys())
        
        # pos_rois = {}
      
        # for roi_id in rois.roi_id.unique():
        #     try:
        #         pos_rois[str(roi_id).zfill(5)] = np.stack([roi_array[(roi_id, frame)] for frame in range(T)])
        #     except KeyError:
        #         break
        #     except ValueError: #Edge case handling for rois very close to the edge, sometimes the intial padding does not work properly due to rounding errors.
        #         roi_size = roi_array[(roi_id, T-1)].shape
        #         pos_rois[str(roi_id).zfill(5)] = np.stack([ip.pad_to_shape(roi_array[(roi_id, frame)], roi_size) for frame in range(T)])
        
        #self.temp_array = pos_rois
        #roi_array_padded = np.stack(roi_array_padded)

        #print('ROIs generated, saving...')
        #self.image_handler.images['spot_images'] = pos_rois
        #self.image_handler.spot_images['spot_images_padded'] = roi_array_padded
            
        #np.savez(self.config['image_path']+os.sep+'spot_images'+self.postfix+'.npz', **roi_array)
        #self.image_handler.images['spot_images'] = image_io.NPZ_wrapper(self.config['image_path']+os.sep+'spot_images'+self.postfix+'.npz')
        #print('ROIs saved.')
        #np.save(self.image_handler.spot_images_path+os.sep+'spot_images_padded.npy', roi_array_padded)

    def gen_roi_imgs_inmem_coursedc(self):
        #TODO: Update to multichannel tracing!
        
        # Use this simplified function if the images that the spots are gathered from are already coursely drift corrected!
        #rois = self.roi_table#.iloc[0:500]
        #imgs = self.
        print('Generating single spot image stacks from coursely drift corrected images.')
        rois = self.image_handler.tables[self.config['spot_input_name']+'_dc_rois']
        for pos, group in tqdm.tqdm(rois.groupby('position')):
            pos_index = self.image_handler.image_lists[self.config['spot_input_name']].index(pos)
            full_image = np.array(self.image_handler.images[self.config['spot_input_name']][pos_index])
            #print(full_image.shape)
            for roi in group.to_dict('records'):
                spot_stack = full_image[:, 
                                roi['ch'], 
                                roi['z_min']:roi['z_max'], 
                                roi['y_min']:roi['y_max'],
                                roi['x_min']:roi['x_max']].copy()
                #print(spot_stack.shape)
                fn = pos+'_'+str(roi['frame'])+'_'+str(roi['roi_id_pos']).zfill(4)
                np.save(self.image_handler.image_save_path+os.sep+'spot_images_dir'+os.sep+fn+'.npy', spot_stack)
        #self.image_handler.images['spot_images'] = all_spots
        if self.image_handler.pos_id is None: #Only do this if not running distributed, otherwise need to collect and zip later.
            image_io.zip_folder(self.image_handler.image_save_path+os.sep+'spot_images_dir', 'spot_images.npz', remove_folder = True)
            self.image_handler.images['spot_images'] = image_io.NPZ_wrapper(self.image_handler.image_save_path+os.sep+'spot_images.npz')
        #np.savez_compressed(self.config['image_path']+os.sep+'spot_images.npz', **all_spots)
        #self.image_handler.images['spot_images'] = image_io.NPZ_wrapper(self.config['image_path']+os.sep+'spot_images.npz')
        

    def fine_dc_single_roi_img(self, roi_img, roi):
        #Shift a single image according to precalculated drifts.
        dz = float(roi['z_px_fine'])
        dy = float(roi['y_px_fine'])
        dx = float(roi['x_px_fine'])
        #roi_image_shifted = delayed(ndi.shift)(roi_image_exp, (dz, dy, dx))
        roi_img = ndi.shift(roi_img, (dz, dy, dx)).astype(np.uint16)
        return roi_img

    def gen_fine_dc_roi_imgs(self):
        #Apply fine scale drift correction to spot images, used mainly for visualizing fits (these images are not used for fitting)
        print('Making fine drift-corrected spot images.')

        imgs = self.image_handler.images['spot_images']
        
        rois = self.image_handler.tables[self.roi_name+'_dc_rois']

        i = 0
        roi_array_fine = []
        for j, frame_stack in tqdm.tqdm(enumerate(imgs)):
            roi_stack_fine = []
            for roi_stack in frame_stack:
                roi_stack_fine.append(self.fine_dc_single_roi_img(roi_stack, rois.iloc[i]))
                i += 1
            roi_array_fine.append(np.stack(roi_stack_fine))

        #roi_imgs_fine = Parallel(n_jobs=-1, verbose=1, prefer='threads')(delayed(self.fine_dc_single_roi_img)(roi_imgs[i], rois.iloc[i]) for i in tqdm(range(roi_imgs.shape[0])))
        #roi_imgs_fine = np.stack(roi_array_fine)
        #roi_array_fine = np.array(roi_array_fine, dtype='object')
        
        self.image_handler.images['spot_images_fine'] = roi_array_fine
        np.savez_compressed(self.image_handler.image_save_path+os.sep+'spot_images_fine.npz', *roi_array_fine)