# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

from looptrace.gaussfit import fitSymmetricGaussian3D
from scipy.ndimage import shift
from looptrace import image_processing_functions as ip
from looptrace import image_io
import os
import numpy as np
import tqdm
import dask.array as da

class Deconvolver:
    '''
    Class for handling generation and detection of e.g. nucleus images.
    '''

    def __init__(self, image_handler):
        self.image_handler = image_handler
        self.config = image_handler.config
        self.pos_list = self.image_handler.image_lists[self.config['decon_input_name']]

    def extract_exp_psf(self):
        '''
        Extract an experimental PDF from a bead image.
        Parameters read from config to segment same beads as used for drift correction.
        Segments and extract beads, filters the intensities to remove possible doublets,
        fits the centers, registers and overlays the beads, calculates an average and normalizes the signal.
        Saves as a .npy ndarray, image is a centered PSF useful for deconvolution.
        '''

        t_slice = self.config['psf_bead_frame']
        ch = self.config['psf_bead_ch']
        threshold = self.config['bead_threshold']
        min_bead_int = self.config['min_bead_intensity']
        n_beads = 500
        try:
            bead_d = self.config['psf_bead_size']
            bead_r = bead_d//2
        except KeyError: #Legacy config
            bead_d = 12
            bead_r = 6

        bead_img = self.image_handler.images[self.config['psf_input_name']][0][t_slice, ch].compute()
        bead_pos = ip.generate_bead_rois(bead_img, threshold, min_bead_int, bead_d, n_beads)
        beads = [ip.extract_single_bead(point, bead_img, bead_roi_px = bead_d) for point in bead_pos]
        bead_ints = np.sum(np.array(beads),axis = (1,2,3))
        perc_high = np.percentile(bead_ints, 40)
        perc_low = np.percentile(bead_ints, 5)
        beads = [b for b in beads if ((np.sum(b) < perc_high) & (np.sum(b) > perc_low))]

        fits = [fitSymmetricGaussian3D(b, 3, [bead_r,bead_r,bead_r]) for b in beads]

        drifts = [np.array([bead_r, bead_r, bead_r])-fit[0][2:5] for fit in fits]
        beads_c = [shift(b, d, mode='wrap') for b,d in zip(beads, drifts)]
        exp_psf = np.mean(np.stack(beads_c), axis=0)[1:,1:,1:]
        exp_psf = (exp_psf-np.min(exp_psf))/(np.max(exp_psf)-np.min(exp_psf))
        np.save(self.image_handler.image_path+os.sep+'exp_psf.npy', exp_psf)
        self.image_handler.images['exp_psf'] = exp_psf
        print('Experimental PSF generated.')


    def decon_seq_images(self):
        #Decovolve images using Flowdec.
        #Using Dask Example from https://github.com/hammerlab/flowdec/blob/master/python/examples/notebooks/Tile-by-tile%20deconvolution%20using%20dask.ipynb

        from flowdec import data as fd_data
        from flowdec.restoration import RichardsonLucyDeconvolver

        decon_ch = self.config['decon_ch']
        if decon_ch is None:
            decon_ch = []
        elif not isinstance(decon_ch, list):
            decon_ch = [decon_ch]
        non_decon_ch = self.config['non_decon_ch']
        if non_decon_ch is None:
            non_decon_ch = []
        elif not isinstance(non_decon_ch, list):
            non_decon_ch = [non_decon_ch]
            
        n_iter = self.config['decon_iter']
        if n_iter == 0:
            print("Iterations set to 0.")
            return   

        try: 
            psf_type = self.config['decon_psf']
        except:
            psf_type = 'gen'

        if psf_type == 'exp':
            try:
                psf =  self.image_handler.images['exp_psf']
                print('Using experimental psf for deconvolution.')
            except KeyError:
                print('Experimental PSF not extracted, extracting now.')
                self.extract_exp_psf()
                psf = self.image_handler.images['exp_psf']
        else:
            from flowdec import psf as fd_psf
            psf = fd_psf.GibsonLanni(size_x=15, size_y=15, size_z=15, pz=0., wavelength=self.config['spot_wavelength']/1000,
                                    na=self.config['objective_na'], res_lateral=self.config['xy_nm']/1000, res_axial=self.config['z_nm']/1000).generate()

        algo = RichardsonLucyDeconvolver(3, pad_mode='2357', pad_min=(8,8,8)).initialize()       
        def run_decon(data, algo, fd_data, psf, n_iter):
            return algo.run(fd_data.Acquisition(data=data, kernel=psf), niter=n_iter).data.astype(np.uint16)
        decon_chunk = lambda chunk: run_decon(data=chunk, algo=algo, fd_data=fd_data, psf=psf, n_iter=n_iter)
        
        for pos in tqdm.tqdm(self.pos_list):
            pos_index = self.image_handler.image_lists[self.config['decon_input_name']].index(pos)
            pos_img = self.image_handler.images[self.config['decon_input_name']][pos_index]
            
            z = image_io.create_zarr_store(path=self.image_handler.image_save_path+os.sep+self.config['decon_input_name']+'_decon',
                    name = self.config['decon_input_name']+'_decon', 
                    pos_name = (pos if pos.endswith('.zarr') else pos+'.zarr'),
                    shape = (pos_img.shape[0],len(decon_ch)+len(non_decon_ch),)+pos_img.shape[-3:], 
                    dtype = np.uint16,  
                    chunks = (1,1,1,pos_img.shape[-2], pos_img.shape[-1]))

            for t in tqdm.tqdm(range(pos_img.shape[0])):
                for i, ch in enumerate(decon_ch):
                    t_img = np.array(pos_img[t,ch])
                    Z,Y,X = t_img.shape

                    if np.any(np.array([Z,Y,X])<5):
                        out = np.zeros_like(t_img)
                    
                    elif (Z>128) or (X>1500) or (Y>1500):
                        if Z <= 128:
                            Z_chunk = Z
                            Z_depth = 0
                        else:
                            Z_chunk = Z//(Z//64)
                            Z_depth = 4
                        if Y <= 512:
                            Y_chunk = Y
                        else:
                            Y_chunk = Y//(Y//512)
                        if X <= 512:
                            X_chunk = X
                        else:
                            X_chunk = X//(X//512)

                        chunk_size = (Z_chunk, Y_chunk, X_chunk)
                        depth = (Z_depth,8,8)
                        arr = da.from_array(t_img, chunks=chunk_size)
                        out = arr.map_overlap(decon_chunk, depth=depth, boundary='reflect', dtype='uint16').compute(num_workers=1)

                    else:
                        out = run_decon(data=t_img, algo=algo, fd_data=fd_data, psf=psf, n_iter=n_iter)
                    print(out.shape)
                    z[t, i] = out.copy()

                for i, ch in enumerate(non_decon_ch):
                    z[t, i + len(decon_ch)] = np.array(pos_img[t,ch])