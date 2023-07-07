#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def FFT2(U):
    # Friendly Fourier transform.
    # np.fft.fft assumes the x axis is from 0 to x; N samples and gives the FFT from freq. 0 to 1/dx, with N samples in between
    # FFT as defined here goes from [-x/2, x/2-dx] to [-f/2, f/2-df]
    # This function assumens an N*N grid, equally sampled in space along x and y
    
    # Discretization in frequency is as below for given dx
    #N = np.shape(U)
    #xy_cells = N[0]
    #fx = fy = 1/(dx*xy_cells)*np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
    #df = 1/(dx*xy_cells)
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(U)))

def iFFT2(A):
    # Friendly inverse Fourier transform.
    # This function assumens an N*N grid, equally sampled in space along fx and fy
    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(A)))

def optimal_cell_size(spot_size,FDFD_dx,min_cells):
    # FFT is very fast for composite numbers, and slow for primes.
    # This function returns the optimal cell size for fast FFT.
    minimum_cell_size = np.max([int((spot_size*1.5/FDFD_dx)),int(min_cells)])
    preferred_cell_sizes = np.genfromtxt('Fast_FFT_lengths.csv', dtype = 'int')     # The values of cell sized for which FFT is fastest
    for i in range(len(preferred_cell_sizes)):
        if preferred_cell_sizes[i] >= minimum_cell_size:
            return preferred_cell_sizes[i]
    
    return minimum_cell_size    # Only happens if minimum_cell_size is greater than 2047. This will be extremely slow on any computer as of 2023.


