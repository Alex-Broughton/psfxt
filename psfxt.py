# Utils
import matplotlib.pyplot as plt
import numpy as np
import numpy
import galsim
from scipy.ndimage import gaussian_filter

def calc_2nd_mom(results):
    # Calculate Second Moments
    e1 = results.observed_shape.e1
    e2 = results.observed_shape.e2
    sigma = results.moments_sigma
    sigma_ave = sigma / (1 - e1**2 - e2**2 )**(0.25) 
    Ixx = (1+e1) * sigma_ave**2
    Iyy = (1-e1) * sigma_ave**2
    Ixy = e2 * (sigma_ave**2) 
    
    return Ixx, Iyy, Ixy

def get_image_stats(img):
    result = galsim.hsm.FindAdaptiveMom(img)
    Ixx, Iyy, Ixy = calc_2nd_mom(result)
    peak = np.max(img.array)
    return [peak, Ixx, Iyy, Ixy]

def simulate(fwhm_func, params):
    
    # Initial Setup
    niter = params['niter']
    imsize = params['imsize']
    source_sigma = params['source_sigma']
    bf_strength = params['bf_strength']
    diffusion_factor = params['diffusion_factor']
    nrecalc = params['nrecalc']
    total_flux = params['total_flux']
    nsteps = params['nsteps']
    t0 = params['t0']
    tf = params['tf']
    image_mean = np.zeros(imsize)
    image_stats = np.zeros((niter,4))
    dt = (tf-t0)/nsteps
    added_flux = total_flux / nsteps
    
    # Check for valid inputs
    assert tf > t0 and t0 >= 0.0
    assert total_flux > 0.0
    assert nsteps > 0
    assert added_flux >= nrecalc
    
    # Generate source
    sensor = galsim.SiliconSensor(strength = bf_strength, diffusion_factor=diffusion_factor,
                                  nrecalc=nrecalc)
    blank_image = galsim.Image(imsize[0],imsize[1],xmin=0,ymin=0,dtype=np.float64)  
    source = galsim.Gaussian(sigma=source_sigma)
    
    image_mean = np.zeros((imsize[0],imsize[1]))
    image_stats = np.zeros((niter, 4))
    
    # Integrate multiple exposures
    for n in range(niter):
        print(f"{n+1}/{niter}")
        
        # Integrate single exposure
        image = blank_image.copy()
        total_added_flux = 0
        time_bins = np.linspace(t0,tf,nsteps+1)
        time_steps = (time_bins[1:] - time_bins[:-1])/2. + time_bins[:-1]
        print(time_steps)

        for t in time_steps:

            # Convolve source with PSF
            fwhm = fwhm_func(t)
            psf = galsim.Gaussian(fwhm=fwhm)
            profile = galsim.Convolve([source, psf]).withFlux(added_flux)

            # Add to image
            image = profile.drawImage(image=image, method="phot", sensor=sensor, 
                                      add_to_image=True, save_photons=True)

            # Next time step
            total_added_flux += added_flux

        # Get single exposure stats
        print(np.sum(image.array))
        image_stats[n] = get_image_stats(image)
        print(image_stats[n])
        image_mean += image.array
        
    image_mean /= niter
    
    final_stats = np.mean(image_stats, axis=0)
    final_stats_errs = np.std(image_stats, axis=0) / np.sqrt(niter)
    
    output = dict()
    output['image_mean'] = image_mean
    output['peak'] = final_stats[0]
    output['peak_err'] = final_stats_errs[0]
    output['ixx'] = final_stats[1]
    output['iyy'] = final_stats[2]
    output['ixy'] = final_stats[3]
    output['ixx_err'] = final_stats_errs[1]
    output['iyy_err'] = final_stats_errs[2]
    output['ixy_err'] = final_stats_errs[3]
    
    print("Done.")
    
    return output

