"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

from flowdec import data as fd_data
from flowdec.restoration import RichardsonLucyDeconvolver
#import tifffile
from looptrace import image_io
import numpy as np
import nd2
import argparse
import dask.array as da

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deconvolve single ND2 image.')
    parser.add_argument("--image_path", help="Image file path")
    parser.add_argument("--psf_path", help="(Optional) Path to experimental PSF for deconvolution.", default=None)
    parser.add_argument('--n_iter', default=30)
    args = parser.parse_args()

    if args.psf_path is not None:
        psf = np.load(args.psf_path)
    else:
        from flowdec import psf as fd_psf
        psf = fd_psf.GibsonLanni(size_x=15, size_y=15, size_z=15, pz=0., wavelength=500/1000,
                                na=1.27, res_lateral=113/1000, res_axial=200/1000).generate()

    algo = RichardsonLucyDeconvolver(3, pad_mode='2357', pad_min=(8,8,8)).initialize()       
    def run_decon(data, algo, fd_data, psf, n_iter):
        return algo.run(fd_data.Acquisition(data=data, kernel=psf), niter=n_iter).data.astype(np.uint16)
    decon_chunk = lambda chunk: run_decon(data=chunk, algo=algo, fd_data=fd_data, psf=psf, n_iter=int(args.n_iter))
    
    img = nd2.ND2File(args.image_path, validate_frames = False).to_dask()
    
    print('Loaded image of shape ', img.shape)
    if img.ndim == 5:
        positions = img.shape[0]
        img = da.moveaxis(img, 1, 2)
        C, Z, Y, X = img.shape[1:]
    else:
        positions = 1
        img = da.moveaxis(img, 0, 1)
        C, Z, Y, X = img.shape

    for pos in range(positions):  
        decon_img = []
        for ch in range(C):
            print('Deconvolving channel ', ch)
            if (Z>128) or (X>1500) or (Y>1500):
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
                arr = da.rechunk(img[pos][ch], chunks=chunk_size)
                out = arr.map_overlap(decon_chunk, depth=depth, boundary='reflect', dtype='uint16').compute(num_workers=1)

            else:
                out = run_decon(data=img[pos][ch], algo=algo, fd_data=fd_data, psf=psf, n_iter=int(args.n_iter))

            decon_img.append(out)
        decon_img = np.stack(decon_img).astype(np.uint16)
        image_io.single_position_to_zarr(decon_img, 
                        path = args.image_path+'_decon',
                        name = 'decon+pos_'+str(pos), 
                        pos_name = 'P'+str(pos).zfill(4), 
                        dtype = np.uint16, 
                        axes=('c','z','y','x'), 
                        chunk_axes = ('y', 'x'), 
                        chunk_split = (1,1),  
                        metadata = None)