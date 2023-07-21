import os
import sys
import numpy as np
import pickle as pkl

sys.path.insert(0,'/sdf/home/a/abrought/simulated/psfxt')
from psfxt import *

def run(params, func, outfile_prefix, sim_num):
    
    # Simulate 
    output = simulate(func, params)
    
    # Write to file
    with open(f'{outfile_prefix}-process-{sim_num}.pkl', 'wb') as handle:
        pkl.dump(output, handle)
    
    return
    
if __name__ == '__main__':
    
    # Fluxes to simulate
    job_id = int(sys.argv[1])
    flux_list = np.arange(2,10+1) * 1.0e5
    
    # Parameters
    params = dict()
    params['niter'] = 1000
    params['imsize'] = (25,25)
    params['source_sigma'] = 0.2
    params['bf_strength'] = 1.0
    params['diffusion_factor'] = 1.0
    params['total_flux'] = flux_list[job_id-1]
    params['nrecalc'] = 100
    params['nsteps'] = 5
    params['t0'] = 0.
    params['tf'] = 15.
    
    # Gaussian PSF(t)
    def fwhm(t):
        t = 15.-t
        # Decays from 1" to 0.75" over 15s
        pixscale = 0.2 # "/px
        fwhm = (1. - .75)/15.*t + .75
        return fwhm / pixscale
    
    outfile_prefix = "./output/small2large"
    
    result = run(params, fwhm, outfile_prefix, sys.argv[1])

