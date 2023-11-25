#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time

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

def FFT_benchmark(max_cell_size = 3000):
    # 2D FFT is one of the main bottlenecks in the simulation. The speed of FFT depends on the size of the matrix non-trivially, and this depends on the CPU too.
    # For example, on an AMD Ryzen 5 3600, 2047*2047 FFT is 3 times faster than 2048*2048. On most other systems, 2048*2048 matrices are faster.
    # I suspect this is due to some SIMD shennanigans.
    # At any rate, FFT speed vs size needs to be benchmarked on each PC before running the simulation to guarantee optimal cell sizes are chosen.

    lower_limit = 250
    sizes = range(lower_limit,max_cell_size+1,2)
    times = np.zeros(len(sizes))
    index = 0
    for FFT_size in sizes:  # Only interested in even cell sizes
        a = np.random.random((FFT_size,FFT_size))+1j*np.random.random((FFT_size,FFT_size))
        start = time.time()
        b = np.fft.fft2(a)
        times[index] = time.time()-start
        index = index + 1
    
    # Find the fastest FFT within 50 consecutive FFT sizes
    batch_size = 50     # keep it even
    optimal_sizes = np.zeros(int(np.ceil(((max_cell_size+1)-lower_limit)/batch_size)),dtype='int')
    for i in range(len(optimal_sizes)):
        optimal_sizes[i] = int(lower_limit+batch_size*i+2*np.argmin(times[int(batch_size*i/2):int(batch_size*(i+1)/2)]))
    
    np.save('optimal_FFT_sizes', optimal_sizes)

def optimal_cell_size(spot_size,FDFD_dx,min_cells):
    # FFT is very fast for composite numbers, and slow for primes.
    # This function returns the optimal cell size for fast FFT.
    minimum_cell_size = np.max([int((spot_size*1.5/FDFD_dx)),int(min_cells)])
    try:
        preferred_cell_sizes = np.load('optimal_FFT_sizes.npy') 
    except FileNotFoundError:
        print("FFT Benchmark file not found. Performing the benchmark...")
        FFT_benchmark()
        print("Benchmark finished!")
        preferred_cell_sizes = np.load('optimal_FFT_sizes.npy') 
    #preferred_cell_sizes = np.genfromtxt('Fast_FFT_lengths.csv', dtype = 'int')     # The values of cell sized for which FFT is fastest
    for i in range(len(preferred_cell_sizes)):
        if preferred_cell_sizes[i] >= minimum_cell_size:
            return int(preferred_cell_sizes[i])     # Python int is variable size, so I don't have to worry about overflow.

    raise ValueError("Maximum allowed value of dx is too small. Increase max_FDFD_dx")
    return minimum_cell_size    # Only reaches this return statement if minimum_cell_size is greater than 2047. This will be extremely slow on any computer as of 2023.

