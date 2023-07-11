# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import looptrace.image_processing_functions as ip
#from looptrace import image_io
from looptrace.gaussfit import fitSymmetricGaussian3D, fitSymmetricGaussian3DMLE
from tqdm import tqdm
#import os

class Tracer:
    def __init__(self, image_handler, roi_name = None):
        '''
        Initialize Tracer class with config read in from YAML file.
    '''
        self.image_handler = image_handler
        self.config_path = image_handler.config_path
        self.config = image_handler.config
        self.drift_table = image_handler.tables[self.config['spot_input_name']+'_drift_correction']
        self.images = self.image_handler.images[self.config['trace_input_name']]
            
        self.pos_list = self.image_handler.image_lists[self.config['spot_input_name']]
        

        if roi_name == None:
            self.roi_name = self.config['spot_input_name']
        else:
            self.roi_name = roi_name

        self.roi_table = image_handler.tables[self.roi_name+'_rois']

        self.all_rois = image_handler.tables[self.roi_name+'_dc_rois']

        self.traces_path = self.image_handler.out_path+self.roi_name+'_traces.csv'

        self.fit_funcs = {'LS': fitSymmetricGaussian3D, 'MLE': fitSymmetricGaussian3DMLE}
        self.fit_func = self.fit_funcs[self.config['fit_func']]


        if self.image_handler.pos_id is not None:
            self.roi_table = self.roi_table[self.roi_table.position.isin(self.pos_list)]
            self.all_rois = self.all_rois[self.all_rois.position.isin(self.pos_list)].reset_index(drop=True)
            self.traces_path = self.image_handler.out_path+self.roi_name+'_traces.csv'[:-4]+'_'+str(self.image_handler.pos_id).zfill(4)+'.csv'
            self.images = self.images[self.roi_table.index.to_list()]


    def trace_single_roi(self, roi_img, mask = None, background = None):
        
        #Fit a single roi with 3D gaussian (MLE or LS as defined in config).
        #Masking by intensity or label image can be used to improve fitting correct spot (set in config)

        if background is not None:
            roi_img = roi_img - background
        if np.any(roi_img) and np.all([d > 2 for d in roi_img.shape]): #Check if empty or too small for fitting
            if mask is None:
                fit = self.fit_func(roi_img, sigma=1, center='max')[0]
            else:
                roi_img_masked = (mask/np.max(mask))**2 * roi_img
                center = list(np.unravel_index(np.argmax(roi_img_masked, axis=None), roi_img.shape))
                fit = self.fit_func(roi_img, sigma=1, center=center)[0]
            return fit
        else:
            fit = np.array([-1, -1, -1, -1, -1, -1, -1])
            return fit
        

    def trace_all_rois(self):
        '''
        Fits 3D gaussian to previously detected ROIs across positions and timeframes.
    
        '''
        imgs = self.images

        fits = []

        #fits = Parallel(n_jobs=-1, prefer='threads')(delayed(self.trace_single_roi)(roi_imgs[i]) for i in tqdm(range(roi_imgs.shape[0])))
        try:
            mask_fits = self.image_handler.config['mask_fits']
        except KeyError:
            mask_fits = False

        try:
            #This only works for a single position at the time currently
            background_frame = self.image_handler.config['substract_background'] #Frame to subtract background from.
            pos_drifts = self.drift_table[self.drift_table.position.isin(self.pos_list)][['z_px_fine', 'y_px_fine', 'x_px_fine']].to_numpy()
            background_rel_drifts = pos_drifts - pos_drifts[background_frame]
        except KeyError:
            background_frame = None

        if mask_fits:
            ref_frames = self.roi_table['frame'].to_list()
            ref_chs = self.roi_table['ch'].to_list()
            for p, pos_imgs in tqdm(enumerate(imgs), total=len(imgs)):
                try:
                    ref_img = pos_imgs[ref_frames[p], ref_chs[p]]
                except IndexError: #Edge case, if images used for tracing (e.g. deconvolved) have fewer channels than the images used for spot detection, channels might be incorrect and defaults to 0 instead.
                    ref_img = pos_imgs[ref_frames[p], 0]
                #print(ref_img.shape)
                for t, frame_img in enumerate(pos_imgs):
                    if frame_img.ndim == 3: #Legacy, only one channel traced and spot images contains no channel axis.
                        if background_frame is not None:
                            frame_img = np.clip(frame_img.astype(np.int16) - ndi.shift(pos_imgs[background_frame], shift = background_rel_drifts[t]), a_min = 0, a_max = None)
                        fits.append(self.trace_single_roi(frame_img, mask = ref_img))
                        
                    elif frame_img.ndim == 4: #Up to date, compatible with single or multichannel tracing.
                        for ch, spot_img in enumerate(frame_img):
                            if background_frame is not None:
                                spot_img = np.clip(spot_img.astype(np.int16) - ndi.shift(pos_imgs[background_frame, ch], shift = background_rel_drifts[t]), a_min = 0, a_max = None)
                            fits.append(self.trace_single_roi(spot_img, mask = ref_img))

                #Parallel(n_jobs=1, prefer='threads')(delayed(self.trace_single_roi)(imgs[p, t], mask= ref_img) for t in range(imgs.shape[1]))

        else:
            for pos_imgs in tqdm(imgs, total=len(imgs)):
                for frame_img in pos_imgs:
                    if frame_img.ndim == 3: #Legacy, only one channel traced and spot images contains no channel axis.
                        fits.append(self.trace_single_roi(frame_img))
                    elif frame_img.ndim == 4: #Up to date, compatible with single or multichannel tracing.
                        for spot_img in frame_img:
                            fits.append(self.trace_single_roi(spot_img))

        trace_res = pd.DataFrame(fits,columns=["BG","A","z_px","y_px","x_px","sigma_z","sigma_xy"])
        #trace_index = pd.DataFrame(fit_rois, columns=["trace_id", "frame", "ref_frame", "position", "drift_z", "drift_y", "drift_x"])
        traces = pd.concat([self.all_rois, trace_res], axis=1)
        traces.rename(columns={"roi_id": "trace_id"}, inplace=True)

        #Apply fine scale drift to fits, and physcial units.
        traces['z_px_dc']=traces['z_px']+traces['z_px_fine']
        traces['y_px_dc']=traces['y_px']+traces['y_px_fine']
        traces['x_px_dc']=traces['x_px']+traces['x_px_fine']
        traces['z_min_glob'] = traces['trace_id'].map(self.roi_table['z_min'].to_dict())
        traces['y_min_glob'] = traces['trace_id'].map(self.roi_table['y_min'].to_dict())
        traces['x_min_glob'] = traces['trace_id'].map(self.roi_table['x_min'].to_dict())
        #traces=traces.drop(columns=['drift_z', 'drift_y', 'drift_x'])
        traces['z']=traces['z_min_glob']+traces['z_px_dc']
        traces['y']=traces['y_min_glob']+traces['y_px_dc']
        traces['x']=traces['x_min_glob']+traces['x_px_dc']
        traces['sigma_z']=traces['sigma_z']
        traces['sigma_xy']=traces['sigma_xy']
        traces = traces.sort_values(['trace_id', 'frame'])


        #self.image_handler.traces = traces
        traces.to_csv(self.traces_path)