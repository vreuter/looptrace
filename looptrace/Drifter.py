# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

#from distutils.command.config import config
#from tkinter import image_names

from dask.delayed import delayed
from joblib.parallel import Parallel
import numpy as np
import pandas as pd
from looptrace import image_processing_functions as ip
from looptrace import image_io
from looptrace.gaussfit import fitSymmetricGaussian3D, fitSymmetricGaussian3DMLE
from skimage.registration import phase_cross_correlation
from skimage.measure import regionprops_table
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from scipy.stats import trim_mean
from joblib import Parallel, delayed
import os
import tqdm
import dask.array as da

class Drifter():

    def __init__(self, image_handler):
        '''
        Initialize Drifter class with config read in from YAML file.
        '''
        self.image_handler = image_handler
        self.config = self.image_handler.config
        self.dc_file_path = self.image_handler.out_path+self.config['reg_input_moving']+'_drift_correction.csv'
        self.images_template = self.image_handler.images[self.config['reg_input_template']]
        self.images_moving = self.image_handler.images[self.config['reg_input_moving']]
        self.full_pos_list = self.image_handler.image_lists[self.config['reg_input_moving']]
        self.pos_list = self.full_pos_list
        
        if self.image_handler.pos_id is not None:
            self.dc_file_path = self.image_handler.out_path+self.config['reg_input_moving']+'_drift_correction.csv'[:-4]+'_'+str(self.image_handler.pos_id).zfill(4)+'.csv'


    def fit_shift_single_bead(self, t_bead, o_bead):
        # Fit the center of two beads using 3D gaussian fit, and fit the shift between the centers.
        if (np.all(t_bead == 0)) or (np.all(o_bead == 0)):
            return np.array([0,0,0])
        try:
            t_fit = np.array(fitSymmetricGaussian3D(t_bead, sigma=1, center=None)[0])
            o_fit = np.array(fitSymmetricGaussian3D(o_bead, sigma=1, center=None)[0])
            shift = t_fit[2:5] - o_fit[2:5]
        except (ValueError, AttributeError) as e:
            print(e, t_fit, o_fit)
            shift = np.array([0,0,0])
        return shift

    def correlate_single_bead(self, t_bead, o_bead, upsampling):
        if (np.all(t_bead == 0)) or (np.all(o_bead == 0)):
            return np.array([0,0,0])
        try:
            shift = phase_cross_correlation(t_bead, o_bead, upsample_factor=upsampling, return_error=False)
        except (ValueError, AttributeError):
            shift = np.array([0,0,0])
        return shift

    def drift_corr(self):
        '''
        Running function for drift correction along T-axis of 6D (PTCZYX) images/arrays.
        Settings set in config file.

        '''

        ref_frame = self.config['reg_ref_frame']
        ch_t = self.config['reg_ch_template']
        ch_o = self.config['reg_ch_moving']
        threshold = self.config['bead_threshold']
        min_bead_int = self.config['min_bead_intensity']
        n_points= self.config['bead_points']
        #dc_bead_img_path = self.config['output_path']+os.sep+'dc_bead_images'
        roi_px = self.config['bead_roi_size']
        ds = self.config['course_drift_downsample']

        try:
            dc_method = self.config['dc_method']
        except KeyError:
            dc_method = 'cc'

        #try:
        #    save_dc_beads = self.config['save_dc_beads']
        #except KeyError:
        #    save_dc_beads = False

        if dc_method == 'course':
            all_drifts=[]
            #out_imgs = []
            for i, pos in enumerate(self.pos_list):
                print(f'Running only course drift correction for position {pos}.')

                if ref_frame == -1:
                    t_img = np.array(self.images_template[i][0, ch_t, ::ds, ::ds, ::ds])
                else:
                    t_img = np.array(self.images_template[i][ref_frame, ch_t, ::ds, ::ds, ::ds])

                for t in tqdm.tqdm(range(self.images_moving[i].shape[0])):
                    if (ref_frame == -1) & (t>0):
                        t_img = np.array(self.images_template[i][t-1, ch_t, ::ds, ::ds, ::ds])
                    o_img = np.array(self.images_moving[i][t, ch_o, ::ds, ::ds, ::ds])
                    drift_course = ip.drift_corr_course(t_img, o_img, downsample=1)

                    drifts = [t,pos]+list(drift_course)+[0,0,0]
                    all_drifts.append(drifts) 
                    print('Drifts:', drifts)
        elif dc_method == 'cc_blocks':
            #Crops blocks (full z-height) from random places, then uses these for fine drift correction.
            all_drifts=[]
            #out_imgs = []
            for i, pos in enumerate(self.pos_list):
                print(f'Running only course drift correction for position {pos}.')
                t_img = np.array(self.images_template[i][ref_frame, ch_t])
                n_blocks = 50
                y_r = np.random.random(n_blocks) * (t_img.shape[-2]*0.8) + 25 #Keep it a bit confined to avoid edges
                x_r = np.random.random(n_blocks) * (t_img.shape[-1]*0.8) + 25 #Kepp it a bit confined to avoid edges
                t_blocks = []
                for j in range(n_blocks):
                    t_blocks.append(t_img[:, int(y_r[j]):int(y_r[j]+20), int(x_r[j]):int(x_r[j]+20)])

                for t in tqdm.tqdm(range(self.images_moving[i].shape[0])):
                    o_img = np.array(self.images_moving[i][t, ch_o])
                    drift_course = ip.drift_corr_course(t_img, o_img, downsample=ds)
                    o_blocks = []
                    for j in range(n_blocks):
                        o_blocks.append(o_img[:, int(y_r[j]-drift_course[-2]):int(y_r[j]+20-drift_course[-2]), int(x_r[j]-drift_course[-1]):int(x_r[j]+20-drift_course[-1])])
                    drift_fine = []
                    for k in range(n_blocks):
                        if t_blocks[k].shape != o_blocks[k].shape:
                            pass
                        else:
                            drift_fine.append(phase_cross_correlation(t_blocks[k], o_blocks[k], upsample_factor=50, return_error=False))
                    drift_fine = trim_mean(np.array(drift_fine), proportiontocut=0.2, axis=0)
                    drift_fine[0] = drift_fine[0] - drift_course[0]
                    drifts = [t,pos]+list(drift_course)+list(drift_fine)
                    all_drifts.append(drifts) 
                    print('Drifts:', drifts)

        else:
            #Run drift correction for each position and save results in table.
            all_drifts=[]
            #out_imgs = []
            for i, pos in enumerate(self.pos_list):
                #pos_imgs = []
                print(f'Running drift correction for position {pos}.')
                
                if ref_frame == -1:
                    t_img = np.array(self.images_template[i][0, ch_t])
                    old_drifts = np.zeros(shape=(self.images_moving[i].shape[0],6))
                else:
                    t_img = np.array(self.images_template[i][ref_frame, ch_t])

                bead_rois = ip.generate_bead_rois(t_img, threshold, min_bead_int, roi_px, n_points)
                #print(bead_rois)
                #t_bead_imgs =  Parallel(n_jobs=-1, prefer='threads')(delayed(ip.extract_single_bead)(point, t_img) for point in bead_rois)
                t_bead_imgs =  [ip.extract_single_bead(point, t_img, bead_roi_px=roi_px) for point in bead_rois]

                for t in tqdm.tqdm(range(self.images_moving[i].shape[0])):
                    if (ref_frame == -1) and (t>0):
                        t_img = np.array(self.images_template[i][t-1, ch_t])
                        bead_rois = ip.generate_bead_rois(t_img, threshold, min_bead_int, roi_px, n_points)
                        t_bead_imgs =  Parallel(n_jobs=-1, prefer='threads')(delayed(ip.extract_single_bead)(point, t_img, bead_roi_px=roi_px) for point in bead_rois)

                    o_img = np.array(self.images_moving[i][t, ch_o])

                    drift_course = ip.drift_corr_course(t_img, o_img, downsample=ds)
                    #drift_course = ip.drift_corr_icp(t_img, o_img, threshold = threshold, min_bead_int = min_bead_int, downsample = ds)

                    #o_bead_imgs = Parallel(n_jobs=-1, prefer='threads')(delayed(ip.extract_single_bead)(point, o_img, drift_course=drift_course) for point in bead_rois)
                    o_bead_imgs = [ip.extract_single_bead(point, o_img, bead_roi_px=roi_px, drift_course=drift_course) for point in bead_rois]
                    if len(bead_rois) > 0:
                        if dc_method == 'cc':
                            drift_fine = Parallel(n_jobs=-1, prefer='threads')(delayed(self.correlate_single_bead)(t_bead, o_bead, 100) 
                                                                                for t_bead, o_bead in zip(t_bead_imgs, o_bead_imgs))
                        elif dc_method == 'fit':
                            drift_fine = Parallel(n_jobs=-1, prefer='threads')(delayed(self.fit_shift_single_bead)(t_bead, o_bead) 
                                                                           for t_bead, o_bead in zip(t_bead_imgs, o_bead_imgs))
                            #drift_fine = [self.fit_shift_single_bead(t_bead, o_bead) for t_bead, o_bead in zip(t_bead_imgs, o_bead_imgs)]
                        else:
                            raise NotImplementedError('Unknown dc method.')       

                        drift_fine = np.array(drift_fine)
                        #print(drift_fine)
                        drift_fine = drift_fine[np.all(drift_fine != 0, axis=1)]
                        if len(drift_fine) < 5: #Usually if out of focus so no beads registered.
                            drift_fine = np.zeros_like(drift_course)
                        else:
                            drift_fine = trim_mean(drift_fine, proportiontocut=0.2, axis=0)
                        
                    else:
                        drift_fine = np.zeros_like(drift_course)

                    if (ref_frame == -1) and (t>0):
                        print('Running sequential dc.')
                        old_drifts[t,:3] = drift_course
                        old_drifts[t,3:] = drift_fine
                        drifts = list(np.sum(old_drifts, axis=0))
                    else:
                        drifts = list(drift_course)+list(drift_fine)

                    all_drifts.append( [t,pos]+drifts) 
                    print('Drifts:', drifts)

                    #if save_dc_beads:
                    #    if not os.path.isdir(dc_bead_img_path):
                    #        os.mkdir(dc_bead_img_path)
                    #    t_bead_imgs = np.stack([ip.pad_to_shape(img, (roi_px, roi_px, roi_px)) for img in t_bead_imgs])
                    #    o_bead_imgs = np.stack([ip.pad_to_shape(img, (roi_px, roi_px, roi_px)) for img in o_bead_imgs])
                        #out_imgs = np.stack([t_bead_imgs, o_bead_imgs])

                        #pos_imgs(dc_bead_img_path+os.sep+pos+'_T'+str(t).zfill(4)+'.npy', out_imgs)

                print('Finished drift correction for position ', pos)
                    
        
        all_drifts=pd.DataFrame(all_drifts, columns=['frame',
                                                    'position',
                                                    'z_px_course',
                                                    'y_px_course',
                                                    'x_px_course',
                                                    'z_px_fine',
                                                    'y_px_fine',
                                                    'x_px_fine',
                                                    ])
        all_drifts['moving_name'] = self.config['reg_input_moving']
        all_drifts['template_name'] = self.config['reg_input_template']
        all_drifts['ref_frame'] = self.config['reg_ref_frame']
        all_drifts['ref_channel'] = self.config['reg_ch_template']
        all_drifts['moving_channel'] = self.config['reg_ch_moving']

        all_drifts.to_csv(self.dc_file_path)
        print('Drift correction complete.')
        self.image_handler.drift_table = all_drifts

    def chrom_shift(self):
        image_name = self.config['chrom_abb_image_name']
        detect_ch = self.config['chrom_abb_detect_channel']
        ref_ch = self.config['chrom_abb_reference_channel']
        mov_chs = self.config['chrom_abb_corr_channels']
        if not isinstance(mov_chs, list):
            mov_chs = [mov_chs]
        frame = self.config['chrom_abb_frame']
        thresh = self.config['chrom_abb_bead_threshold']
        roi_px = self.config['chrom_abb_bead_size']

        all_fits = []
        for pos, img in enumerate(self.image_handler.images[image_name]):
            img = np.array(img[frame])
            print('Loaded image from position ', pos)
            peaks = peak_local_max(img[detect_ch], min_distance=8, threshold_abs=thresh)
            print('Found ', len(peaks), ' peaks. Fitting...')
            for ch in [ref_ch]+mov_chs:
                for i, peak in enumerate(peaks):
                    bead = ip.extract_single_bead(peak, img[ch], bead_roi_px=roi_px)
                    fit = fitSymmetricGaussian3D(bead, sigma=1,  center = None)[0]
                    fit_global = fit[2:5] + peak - roi_px//2
                    all_fits.append([str(pos)+'_'+str(i), ch]+list(fit)+list(fit_global))
        all_fits = pd.DataFrame(all_fits, columns=['roi_id', 'ch', 'BG', 'A', 'z_loc', 'y_loc', 'x_loc', 'sigma_z', 'sigma_xz', 'z', 'y', 'x'])
        all_fits['A_to_BG'] = all_fits['A']/all_fits['BG']
        all_fits = all_fits.groupby('roi_id').filter(lambda x: x['A_to_BG'].min()>5)
        all_fits.reset_index(drop=True, inplace=True)
        #all_fits.to_csv(self.image_handler.out_path+image_name+'_frame_'+str(frame)+'_ref_ch_'+str(ref_ch)+'_mov_chs_'+str(mov_chs)+'_chrom_abb_fits.csv')
        #all_fits = all_fits[(all_fits.A/all_fits.BG) > 5]
        affines = []
        points_template = all_fits[all_fits.ch == int(ref_ch)][['z', 'y', 'x']].values
        for ch in mov_chs:
            points_moving = all_fits[all_fits.ch == int(ch)][['z', 'y', 'x']].values
            #print(points_template, points_moving)
            affine_matrix = ip.least_squares_transform(points_template, points_moving)
            affines.append(affine_matrix)
        affines = np.stack(affines)
        print(affines)
        np.save(self.image_handler.out_path+image_name+'_frame_'+str(frame)+'_ref_ch_'+str(ref_ch)+'_mov_chs_'+str(mov_chs)+'_affine.npy', affines)

#p =np.concatenate(rois, axis=0)

    def gen_dc_images(self, pos):
        '''
        Makes internal coursly drift corrected images based on precalculated drift
        correction (see Drifter class for details).
        '''
        n_t = self.images[0].shape[0]
        pos_index = self.pos_list.index(pos)
        pos_img = []
        for t in range(n_t):
            shift = tuple(self.drift_table.query('position == @pos').iloc[t][['z_px_course', 'y_px_course', 'x_px_course']])
            pos_img.append(da.roll(self.images[pos_index][t], shift = shift, axis = (1,2,3)))
        self.dc_images = da.stack(pos_img)

        print('DC images generated.')

    def save_proj_dc_images(self):
        '''
        Makes internal coursly drift corrected images based on precalculated drift
        correction (see Drifter class for details).
        '''

        for i, pos in tqdm.tqdm(enumerate(self.pos_list)):
            #pos_index = self.full_pos_list.index(pos)
            #pos_img = self.images_moving[pos_index]
            proj_img = da.max(self.images_moving[i], axis=2)
            z = image_io.create_zarr_store(path=self.image_handler.image_save_path+os.sep+self.config['reg_input_moving']+'_max_proj_dc',
                                            name = self.config['reg_input_moving']+'_max_proj_dc', 
                                            pos_name = pos,
                                            shape = proj_img.shape, 
                                            dtype = np.uint16,  
                                            chunks = (1,1,proj_img.shape[-2], proj_img.shape[-1]))

            n_t = proj_img.shape[0]
            
            for t in tqdm.tqdm(range(n_t)):
                shift = self.image_handler.tables[self.config['reg_input_moving']+'_drift_correction'].query('(position == @pos) & (frame == @t)')[['y_px_course', 'x_px_course', 'y_px_fine', 'x_px_fine']].to_numpy()[0]
                shift = (shift[0]+shift[2], shift[1]+shift[3])
                z[t] = ndi.shift(proj_img[t].compute(), shift=(0,)+shift, order = 1)
        
        print('DC images generated.')
    
    def save_dc_images(self):
        '''
        Makes internal coursly drift corrected images based on precalculated drift
        correction (see Drifter class for details).
        
        P, T, C, Z, Y, X = self.images.shape
        compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
        chunks = (1,1,1,Y,X)
        
        if not os.path.isdir(self.maxz_dc_folder):
            os.mkdir(self.maxz_dc_folder)
        '''

        for i, pos in tqdm.tqdm(enumerate(self.pos_list)):
            #pos_index = self.full_pos_list.index(pos)
            #pos_img = self.images_moving[pos_index]
            img = self.images_moving[i]
            z = image_io.create_zarr_store(path=self.image_handler.image_save_path+os.sep+self.config['reg_input_moving']+'_dc',
                                            name = self.config['reg_input_moving']+'_dc', 
                                            pos_name = pos,
                                            shape = img.shape, 
                                            dtype = np.uint16,  
                                            chunks = (1,1,1,img.shape[-2], img.shape[-1]))

            n_t = img.shape[0]
            
            for t in tqdm.tqdm(range(n_t)):
                shift = self.image_handler.tables[self.config['reg_input_moving']+'_drift_correction'].query('position == @pos').iloc[t][['z_px_course','y_px_course','x_px_course', 'z_px_fine', 'y_px_fine', 'x_px_fine']]
                shift = (shift[0]+shift[3], shift[1]+shift[4], shift[2]+shift[5])
                z[t] = ndi.shift(img[t].compute(), shift=(0,)+shift, order = 0)
        
        print('DC images generated.')