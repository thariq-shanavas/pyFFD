#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numba import jit, njit
from rocket_fft import numpy_like, scipy_like

numpy_like()

#@jit(forceobj=True)
@njit
def FFT2(U):
    # Friendly Fourier transform.
    # np.fft.fft assumes the x axis is from 0 to x; N samples and gives the FFT from freq. 0 to 1/dx, with N samples in between
    # FFT as defined here goes from [-x/2, x/2-dx] to [-f/2, f/2-df]
    # For max. efficiency, set N as a power of 2.
    # This function assumens an N*N grid, equally sampled in space along x and y
    
    # Discretization in frequency is as below for given dx
    #N = np.shape(U)
    #xy_cells = N[0]
    #fx = fy = 1/(dx*xy_cells)*np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
    #df = 1/(dx*xy_cells)
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(U)))

#@jit(forceobj=True)
@njit
def iFFT2(A):
    # Friendly inverse Fourier transform.
    # For max. efficiency, set N as a power of 2.
    # This function assumens an N*N grid, equally sampled in space along fx and fy
    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(A)))
