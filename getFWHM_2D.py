import numpy as np
import scipy.optimize as opt

def twoD_GaussianScaledAmp(xy_cells, xo, yo, sigma, amplitude):
    #Function to fit, returns 2D gaussian function as 1D array
    x = y = np.linspace(0, xy_cells, xy_cells)    
    x, y = np.meshgrid(x, y)
    xo = float(xo)
    yo = float(yo)    
    g = amplitude*np.exp( - (((x-xo)**2)/(2*sigma**2) + ((y-yo)**2)/(2*sigma**2)))
    return g.ravel()

def getFWHM_GaussianFitScaledAmp(img):
    #Returns: FWHMs in pixels
    
    xy_cells = img.shape[0]
    initial_guess = (xy_cells/2,xy_cells/2,10,1)
    img_scaled = np.clip((img) / (img.max()),0,1)
    kwargs = {"ftol":1e-5,"gtol":1e-5}
    popt, pcov = opt.curve_fit(twoD_GaussianScaledAmp, xy_cells, 
                               img_scaled.ravel(), p0=initial_guess,
                               # Bounds for center_x,   center_y,     sigma,    amplitude,    offset
                               bounds = ((xy_cells*0.45, xy_cells*0.45, 1,          0.8),
                                         (xy_cells*0.55, xy_cells*0.55, 100,         1.2)), method = 'trf', **kwargs)
    xcenter, ycenter, sigma, amp = popt[0], popt[1], popt[2], popt[3]
    FWHM = np.abs(4*sigma*np.sqrt(-0.5*np.log(0.5)))
    
    return FWHM