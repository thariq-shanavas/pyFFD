# -*- coding: utf-8 -*-

import numpy as np
import scipy.special as special

def LG_OAM_beam(xy_cells,dx,beam_radius,l):
    
    indices = np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
    xx, yy = np.meshgrid(dx*indices,dx*indices)
    r = np.sqrt(xx**2+yy**2) # Radial coordinate
    beam = np.exp(-(xx**2+yy**2)/(beam_radius**2))
    beam = beam*np.exp(1j*l*np.arctan2(yy,xx))*(np.sqrt(2)*r/beam_radius)**l   # Apply spiral phase
    beam = beam/np.sqrt(np.sum(np.abs(beam)**2*dx**2))  # Normalization

    return beam


def HG_beam(xy_cells,dx,beam_radius,u,v):
    
    indices = np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
    xx, yy = np.meshgrid(dx*indices,dx*indices)
    #r = np.sqrt(xx**2+yy**2) # Radial coordinate
    beam = np.exp(-(xx**2+yy**2)/(beam_radius**2))  # Gaussian envelope
    Hx = special.hermite(u)
    Hy = special.hermite(v)
    beam = beam*Hx(np.sqrt(2)*xx/beam_radius)*Hy(np.sqrt(2)*yy/beam_radius)   # Apply spiral phase
    beam = beam/np.sqrt(np.sum(np.abs(beam)**2*dx**2))  # Normalization
    
    return beam

def Gaussian_beam(xy_cells,dx,beam_radius):
    
    # beam_radius is 1/e*2 intensity point. Note that this function returns electric field, not intensity.
    indices = np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
    xx, yy = np.meshgrid(dx*indices,dx*indices)
    #r = np.sqrt(xx**2+yy**2) # Radial coordinate
    beam = np.exp(-(xx**2+yy**2)/(beam_radius**2))  # Gaussian envelope
    beam = beam/np.sqrt(np.sum(np.abs(beam)**2*dx**2))  # Normalization
    
    return beam