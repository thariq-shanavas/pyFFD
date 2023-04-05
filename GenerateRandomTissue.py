# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 09:59:59 2022

@author: thariq
"""

import numpy as np
import scipy.signal as signal

def RandomTissue(xy_cells, Total_steps, wavelength, dx, dz, n_h, ls, g, unique_layers):
    # Generates random fluctuations in refractive index following https://doi.org/10.1364/OL.44.004989
    
    
    
    if g<0.8 or g>0.98:
        raise NameError('g outside range (0.8,0.98)')
    sigma_p = np.sqrt(dz/ls) # Std. deviation of phase from n_ih along distance dz
    g_vs_sigma_x = np.genfromtxt('g_vs_sigmaX.csv', delimiter=',')
    sigma_x = np.interp(g,g_vs_sigma_x[:,1],g_vs_sigma_x[:,0])*wavelength
    if sigma_x < dx:
        print('Warning: fluctuations in tissue index finer than transverse resolution')
    n_ih = np.zeros((xy_cells,xy_cells,unique_layers),dtype=np.float_)
    indices = np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
    xx, yy = np.meshgrid(dx*indices,dx*indices)
    mask = 1/(2*np.pi*sigma_x**2)*np.exp(-(xx**2+yy**2)/(2*sigma_x**2))
    mask = mask/np.sum(mask*dx**2)  # Normalization
    for i in range(0,unique_layers):
        n_ih[:,:,i] = (wavelength/(2*np.pi*dz))*np.random.normal(scale = sigma_p, size=(xy_cells,xy_cells))
        n_ih[:,:,i] = signal.fftconvolve(n_ih[:,:,i],mask,mode='same')*dx**2
    
    return n_h+n_ih
