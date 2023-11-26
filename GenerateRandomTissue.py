# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 09:59:59 2022

@author: thariq
"""

import numpy as np
import scipy.signal as signal
from scipy.stats import norm

def RandomTissue(args):
    [xy_cells, wavelength, dx, dz, n_h, ls, g, unique_layers, random_seed] = args
    # Generates random fluctuations in refractive index following https://doi.org/10.1364/OL.44.004989
    print('Generating tissue with seed %1.0f, cell size %1.0f and cell width %1.0f um' %(random_seed,xy_cells,xy_cells*dx*10**6))
    rand = np.random.default_rng(random_seed)
    
    if g<0.8 or g>0.98:
        raise NameError('g outside range (0.8,0.98)')
    sigma_p = np.sqrt(dz/ls) # Std. deviation of phase from n_ih along distance dz
    g_vs_sigma_x = np.genfromtxt('g_vs_sigmaX.csv', delimiter=',')
    sigma_x = np.interp(g,g_vs_sigma_x[:,1],g_vs_sigma_x[:,0])*wavelength
    if sigma_x < 1.5*dx:
        print('Warning: fluctuations in tissue index finer than transverse resolution')
    d_phi = np.zeros((xy_cells,xy_cells,unique_layers),dtype=np.float32)
    indices = np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
    xx, yy = np.meshgrid(dx*indices,dx*indices)
    mask = np.exp(-(xx**2+yy**2)/(2*sigma_x**2))

    for i in range(0,unique_layers):
        d_phi[:,:,i] = rand.normal(scale = sigma_p, size=(xy_cells,xy_cells))
        d_phi[:,:,i] = signal.fftconvolve(d_phi[:,:,i],mask,mode='same')

    d_phi = d_phi*sigma_p/norm.fit(d_phi.flatten())[1]  # Normalizing to make sure phase profile has a std. deviation of sigma_p after the convolution
    n_ih = d_phi*wavelength/(dz*2*np.pi) # Convert phase mask into refractive index fluctuation
    return n_h+n_ih