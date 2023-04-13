#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from FriendlyFourierTransform import FFT2, iFFT2

def propagate(U, A, distance, current_step, dx, dz, xy_cells, index,imaging_depth_indices, absorption_padding, Absorption_strength, wavelength):
    
    # Suppressing the evanescent field to ensure stability of the algorithm. Unclear what the consequences are.
    # Inputs
    # U: xy_cells*xy_cells*3 complex
    # Electric field in real space. Modifies U[:,:,2]
    
    # A: xy_cells*xy_cells*3 complex
    # In fourier space along x,y. Real space along z
    
    # distance, meters float
    
    # current_step: int
    # Where the simulation is along z
    
    # dx: float meters
    # Assumes dx = dy

    # dz: float meters

    # num_xy: int dvisions in x = divisions in y
    
    # index: xy_cells*xy_cells*z_cells complex
    # refractive index
    
    # Function modifies the value of Field at index current_step.
    # current_step, int >= 2
    
    # Returns
    # Field
    # current_step
    # Field(:,:,1) is duplicated as Field(:,:,1)
    print('Solving Helmholtz equation...')
    #if dx<wavelength/(np.sqrt(2)*(np.average(index)+0.1)):
    #    raise NameError('Transverse resolution too small. Algorithm is numerically unstable.')
        
                          
    indices = np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
    #kxkx, kyky = np.meshgrid(dk*indices,dk*indices)
    f = 1/(dx*xy_cells)*indices
    k = 2*np.pi*index/wavelength
    k0 = 2*np.pi/wavelength
    steps = int(distance/dz)
    fxfx,fyfy = np.meshgrid(f,f)
    Field_snapshots = np.zeros((xy_cells,xy_cells,1+np.size(imaging_depth_indices)),dtype=np.complex64)
    Field_snapshots_index = 1   # First snapshot is seed
    
    # Absorption boundary condition profile = complex part of index
    xy_range = dx*xy_cells/2
    xx, yy = np.meshgrid(dx*indices,dx*indices)
    absorption_profile = 1j*Absorption_strength*(np.exp((np.abs(xx)-xy_range)/absorption_padding) + np.exp((np.abs(yy)-xy_range)/absorption_padding))
    unique_layers = np.shape(index)[2]
    
    for i in range(current_step,current_step+steps):
        # i  : z+dz
        # i-1: z
        # i-2: z-dz
        
        # Some scaling issues remain
        #A[:,:,i] = dz**2*(4*np.pi**2*(fxfx**2+fyfy**2))*A[:,:,i-1] - dz**2*FFT2(k**2*U[:,:,i-1],dx)[0] + 2*A[:,:,i-1] - A[:,:,i-2]
        A[:,:,2] = dz**2*(4*np.pi**2*(fxfx**2+fyfy**2))*A[:,:,1] - dz**2*FFT2(k[:,:,i%unique_layers]**2*U[:,:,1]) + 2*A[:,:,1] - A[:,:,0]
        #A[:,:,2] = A[:,:,2]*((fxfx**2+fyfy**2)<(1/wavelength)**2).astype(float)
        A[:,:,2][(fxfx**2+fyfy**2)>(1/wavelength)**2] = 0
        
        # Making a paraxial approximation here. i.e., assuming k_z ~ k. This is not a big deal since absorption_profile is zero everywhere except
        # the boundary of the simulation volume in real space anyway.
        U[:,:,2] = iFFT2(A[:,:,2]) * np.exp(1j*k0*dz*absorption_profile)
        A[:,:,2] = FFT2(U[:,:,2])
        
        A[:,:,0] = A[:,:,1]
        A[:,:,1] = A[:,:,2]
        U[:,:,0] = U[:,:,1]
        U[:,:,1] = U[:,:,2]
        
        current_step = current_step + 1
    
        if current_step in imaging_depth_indices:
            Field_snapshots[:,:,Field_snapshots_index] = U[:,:,1]
            Field_snapshots_index = Field_snapshots_index+1
            print('Simulation at',int(current_step*dz*10**6),'um')
    
    
    
    
    return U, A, Field_snapshots, current_step
    

def propagate_Fourier(U, A, distance, current_step, dx, dz, xy_cells, index,imaging_depth_indices, absorption_padding, Absorption_strength, wavelength, suppress_evanescent = False):
    #(U, A, distance, current_step, dx, dz, xy_cells, index,imaging_depth_indices):
    
    # Inputs
    # U: xy_cells*xy_cells*3 complex
    # Electric field in real space. Modifies U[:,:,2]
    
    # A: xy_cells*xy_cells*3 complex
    # In fourier space along x,y. Real space along z
    
    # distance, meters float
    
    # current_step: int
    # Where the simulation is along z
    
    # dx: float meters
    # Assumes dx = dy

    # dz: float meters

    # num_xy: int dvisions in x = divisions in y
    
    # index: xy_cells*xy_cells*z_cells complex
    # refractive index
    
    # Function modifies the value of Field at index current_step.
    # current_step, int >= 2
    
    # Returns
    # Field
    # current_step
    # Field(:,:,1) is duplicated as Field(:,:,1)
    print('Using standard split-step method...')
    n_h = np.average(index)
    n_ih = index - n_h
    dk = 2*np.pi/(dx*xy_cells)
    indices = np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
    kxkx, kyky = np.meshgrid(dk*indices,dk*indices)
    f = 1/(dx*xy_cells)*indices
    k = 2*np.pi*n_h/wavelength
    k0 = 2*np.pi/wavelength
    H = np.exp(1j*dz*np.emath.sqrt((k)**2-kxkx**2-kyky**2))
    steps = int(distance/dz)
    fxfx,fyfy = np.meshgrid(f,f)
    if suppress_evanescent:
        mask = ((fxfx**2+fyfy**2)<(1/wavelength)**2).astype(float)
    else:
        mask = 1
    Field_snapshots = np.zeros((xy_cells,xy_cells,1+np.size(imaging_depth_indices)),dtype=np.complex64)
    Field_snapshots_index = 1   # First snapshot is U[:,:,0]
    
    # Absorption boundary condition profile = complex part of index
    xy_range = dx*xy_cells/2
    xx, yy = np.meshgrid(dx*indices,dx*indices)
    absorption_profile = 1j*Absorption_strength*(np.exp((np.abs(xx)-xy_range)/absorption_padding) + np.exp((np.abs(yy)-xy_range)/absorption_padding))
    unique_layers = np.shape(index)[2]
    
    for i in range(current_step,current_step+steps):
        # i  : z+dz
        # i-1: z
        # i-2: z-dz
        U[:,:,1] = iFFT2(FFT2(mask*U[:,:,0]*np.exp(1j*k0*(n_ih[:,:,i%unique_layers]+absorption_profile)*dz))*H)        
        U[:,:,0] = U[:,:,1]
        current_step = current_step + 1
    
        if current_step in imaging_depth_indices:
            Field_snapshots[:,:,Field_snapshots_index] = U[:,:,1]
            Field_snapshots_index = Field_snapshots_index+1
            print('Simulation at',int(current_step*dz*10**6),'um')
    
    
    
    
    return U, A, Field_snapshots, current_step



def propagate_FiniteDifference(U, A, distance, current_step, dx, dz, xy_cells, index,imaging_depth_indices, absorption_padding, Absorption_strength, wavelength):
    
    # This has the same problem: If dx < lambda/sqrt(2), the field blows up.
    # Inputs
    # U: xy_cells*xy_cells*3 complex
    # Electric field in real space. Does not read U[:,:,2] and modifies it in the first step
    
    # A: xy_cells*xy_cells*3 complex
    # In fourier space along x,y. Real space along z
    # Unecessary for this algorithm, but we'll keep it to maintain the same structure as others
    
    # distance, meters float
    
    # current_step: int
    # Where the simulation is along z
    
    # dx: float meters
    # Assumes dx = dy

    # dz: float meters

    # xy_cells: int dvisions in x = divisions in y
    
    # index: xy_cells*xy_cells*unique_layers complex
    # refractive index
        
    # Function modifies the value of Field at index current_step.
    # current_step, int >= 2
    
    # Returns
    # Field
    # current_step
    # Field(:,:,1) is duplicated as Field(:,:,2)
    print('Solving Helmholtz equation by finite difference...')

        
                          
    indices = np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
    #kxkx, kyky = np.meshgrid(dk*indices,dk*indices)
    f = 1/(dx*xy_cells)*indices
    k = 2*np.pi*index/wavelength
    k0 = 2*np.pi/wavelength
    steps = int(distance/dz)
    fxfx,fyfy = np.meshgrid(f,f)
    Field_snapshots = np.zeros((xy_cells,xy_cells,1+np.size(imaging_depth_indices)),dtype=np.complex64)
    Field_snapshots_index = 1   # First snapshot is seed
    
    # Absorption boundary condition profile = complex part of index
    xy_range = dx*xy_cells/2
    xx, yy = np.meshgrid(dx*indices,dx*indices)
    absorption_profile = 1j*Absorption_strength*(np.exp((np.abs(xx)-xy_range)/absorption_padding) + np.exp((np.abs(yy)-xy_range)/absorption_padding))
    # Cannot use imaginary index because it makes the equation numerically unstable. If necessary, apply this profile after U[:,:,2] is calculated as
    # U[:,:,2] = exp(-alpha*dz)*U[:,:,2]
    unique_layers = np.shape(index)[2]
    dz_dx = dz/dx
    
    for i in range(current_step,current_step+steps):
        # I'm not sure whether ax=0 or ax=1 is x derivative, but we're adding the two anyway, so...
        d2Udx2 = (np.roll(U[:,:,1],1,axis=0)+np.roll(U[:,:,1],-1,axis=0)-2*U[:,:,1])
        d2Udy2 = (np.roll(U[:,:,1],1,axis=1)+np.roll(U[:,:,1],-1,axis=1)-2*U[:,:,1])
        #print(current_step)
        U[:,:,2] = 2*U[:,:,1]-U[:,:,0]-(dz_dx**2)*(d2Udx2+d2Udy2)-(dz*k0*index[:,:,i%unique_layers])**2*U[:,:,1]
        U[:,:,0] = U[:,:,1]
        U[:,:,1] = U[:,:,2]
        
        current_step = current_step + 1
    
        if current_step in imaging_depth_indices:
            Field_snapshots[:,:,Field_snapshots_index] = U[:,:,1]
            Field_snapshots_index = Field_snapshots_index+1
            print('Simulation at',int(current_step*dz*10**6),'um')
    
    
    
    
    return U, A, Field_snapshots, current_step
