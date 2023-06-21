#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from FriendlyFourierTransform import FFT2, iFFT2
#import matplotlib.pyplot as plt
from numba import njit, objmode, types
# Uses Rocket-FFT

# Numba does not support numpy.meshgrid yet, so here we are
@njit
def meshgrid(x, y):
    xx = np.empty(shape=(x.size, y.size), dtype=x.dtype)
    yy = np.empty(shape=(x.size, y.size), dtype=y.dtype)
    for j in range(y.size):
        for k in range(x.size):
            xx[j,k] = x[k]  # change to x[k] if indexing xy
            yy[j,k] = y[j]  # change to y[j] if indexing xy            
    return xx, yy

'''
out_type = types.complex64[:,:]
@jit(nopython=True)
def roll(a, shift, axis):
    with objmode(out=out_type):
        out = np.roll(a, shift, axis)
    return out
'''
'''
@jit(nopython=True)
def roll(a,shift,axis):
    size = a.shape[0]
    b = np.zeros_like(a)
    if shift == 1 and axis == 0:
        b[1:,:] = a[:-1,:]
    elif shift == -1 and axis == 0:
        b[:-1,:] = a[1:,:]
    elif shift == 1 and axis == 1:
        b[:,1:] = a[:,:-1]
    elif shift == -1 and axis == 1:
        b[:,:-1] = a[:,1:]
    return b
'''
@njit
def roll(a,shift,axis):
    size = a.shape[0]
    if axis == 1:
        z = np.zeros((a.shape[0],1)).astype(a.dtype)
    else:
        z = np.zeros((1,a.shape[0])).astype(a.dtype)
    if shift == 1 and axis == 0:
        return np.concatenate((z,a[:-1,:]),axis)
    elif shift == -1 and axis == 0:
        return np.concatenate((a[1:,:],z),axis)
    elif shift == 1 and axis == 1:
        return np.concatenate((z,a[:,:-1]),axis)
    elif shift == -1 and axis == 1:
        return np.concatenate((a[:,1:],z),axis)



'''
def roll(a, shift, axis):
    # Strictly onle for 3 dimensional numpy arrays
    b = np.empty_like(a)
    (axis0_size,axis1_size,axis2_size) = a.shape
    if axis == 0:
        for i in np.range(axis0_size):

'''


def propagate(U, A, distance, current_step, dx, dz, xy_cells, index,imaging_depth_indices, absorption_padding, Absorption_strength, wavelength, suppress_evanescent = True):
    
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
    absorption_profile_real_space = np.exp(1j*k0*dz*absorption_profile)

    for i in range(current_step,current_step+steps):
        # i  : z+dz
        # i-1: z
        # i-2: z-dz
        
        # Some scaling issues remain
        #A[:,:,i] = dz**2*(4*np.pi**2*(fxfx**2+fyfy**2))*A[:,:,i-1] - dz**2*FFT2(k**2*U[:,:,i-1],dx)[0] + 2*A[:,:,i-1] - A[:,:,i-2]
        A[:,:,2] = dz**2*(4*np.pi**2*(fxfx**2+fyfy**2))*A[:,:,1] - dz**2*FFT2(k[:,:,i%unique_layers]**2*U[:,:,1]) + 2*A[:,:,1] - A[:,:,0]
        #A[:,:,2] = A[:,:,2]*((fxfx**2+fyfy**2)<(1/wavelength)**2).astype(float)
        if suppress_evanescent:
            A[:,:,2][(fxfx**2+fyfy**2)>(1/wavelength)**2] = 0
        
        # Making a paraxial approximation here. i.e., assuming k_z ~ k. This is not a big deal since absorption_profile is zero everywhere except
        # the boundary of the simulation volume in real space anyway.
        U[:,:,2] = iFFT2(A[:,:,2]) * absorption_profile_real_space
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
        #plt.pcolormesh(mask)
        #plt.colorbar()
        #plt.title('Fourier plane mask')
        #plt.show()
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
        U[:,:,1] = iFFT2(mask*FFT2(U[:,:,0]*np.exp(1j*k0*(n_ih[:,:,i%unique_layers]+absorption_profile)*dz))*H)        
        U[:,:,0] = U[:,:,1]
        current_step = current_step + 1
    
        if current_step in imaging_depth_indices:
            Field_snapshots[:,:,Field_snapshots_index] = U[:,:,1]
            Field_snapshots_index = Field_snapshots_index+1
            print('Simulation at',int(current_step*dz*10**6),'um')
    
    
    
    
    return U, A, Field_snapshots, current_step


def propagate_FiniteDifference(U, A, distance, current_step, dx, dz, xy_cells, index,imaging_depth_indices, absorption_padding, Absorption_strength, wavelength, suppress_evanescent = False):
    
    # TODO: Numba, numexpr to speed up computation

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
    # print('Solving Helmholtz equation by finite difference...')
    indices = np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
    #kxkx, kyky = np.meshgrid(dk*indices,dk*indices)
    f = 1/(dx*xy_cells)*indices
    k0 = 2*np.pi/wavelength
    steps = int(distance/dz)
    fxfx,fyfy = np.meshgrid(f,f)
    Field_snapshots = np.zeros((xy_cells,xy_cells,1+np.size(imaging_depth_indices)),dtype=np.complex64)
    
    
    # Absorption boundary condition profile = complex part of index
    xy_range = dx*xy_cells/2
    xx, yy = np.meshgrid(dx*indices,dx*indices)
    absorption_profile = np.exp((-dz*Absorption_strength*(np.exp((np.abs(xx)-xy_range)/absorption_padding) + np.exp((np.abs(yy)-xy_range)/absorption_padding))))
    # Cannot use imaginary index because it makes the equation numerically unstable. If necessary, apply this profile after U[:,:,2] is calculated as
    # U[:,:,2] = exp(-alpha*dz)*U[:,:,2]
    unique_layers = np.shape(index)[2]
    dz_dx = dz/dx
    
    # Masking evanescent fields
    
    f = 1/(dx*xy_cells)*indices
    fxfx,fyfy = np.meshgrid(f,f)
    mask = ((fxfx**2+fyfy**2)<(1/wavelength)**2).astype(float)

    for i in range(current_step,current_step+steps):
        # I'm not sure whether ax=0 or ax=1 is x derivative, but we're adding the two anyway, so...
        d2Udx2 = (np.roll(U[:,:,1],1,axis=0)+np.roll(U[:,:,1],-1,axis=0)-2*U[:,:,1])
        d2Udy2 = (np.roll(U[:,:,1],1,axis=1)+np.roll(U[:,:,1],-1,axis=1)-2*U[:,:,1])
        U[:,:,2] = absorption_profile*(2*U[:,:,1]-U[:,:,0]-(dz_dx**2)*(d2Udx2+d2Udy2)-(dz*k0*index[:,:,i%unique_layers])**2*U[:,:,1])
        
        if suppress_evanescent:
            U[:,:,2] =  iFFT2(mask*FFT2(U[:,:,2]))

        U[:,:,0] = U[:,:,1]
        U[:,:,1] = U[:,:,2]
        
        current_step = current_step + 1
    
    
    
    
    return U, A, Field_snapshots, current_step

@njit
def Vector_FiniteDifference(Ux, Uy, Uz, distance, dx, dz, xy_cells, index, wavelength, suppress_evanescent):
    
    # This has the same problem: If dx < lambda/sqrt(2), the field blows up.
    # Implementation follows Eq. 3-8 in Goodman Fourier Optics

    # x axis is the second index (axis=1). y axis is first (axis=0)
    # This makes sure pcolor represents fields accurately

    indices = np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells).astype(np.int32)
    f = 1/(dx*xy_cells)*indices
    k0 = 2*np.pi/wavelength
    steps = int(distance/dz)
    fxfx,fyfy = meshgrid(f,f)
    
    
    # Absorption boundary condition profile = complex part of index
    xy_range = dx*xy_cells/2
    xx, yy = meshgrid(dx*indices,dx*indices)
    absorption_profile = np.exp(-(np.exp(0.1*(np.abs(xx)/dx-xy_cells/2+10)) + np.exp(0.1*(np.abs(yy)/dx-xy_cells/2+10))))
    # Cannot use imaginary index because it makes the equation numerically unstable. If necessary, apply this profile after U[:,:,2] is calculated as
    # U[:,:,2] = exp(-alpha*dz)*U[:,:,2]
    unique_layers = np.shape(index)[2]
    dz_dx = dz/dx
    
    # Masking evanescent fields
    f = 1/(dx*xy_cells)*indices
    fxfx,fyfy = meshgrid(f,f)
    #mask = ((fxfx**2+fyfy**2)<(1/wavelength)**2).astype(float)

    ## Type 2 Fourier mask
    alpha = 2*np.pi*np.sqrt(np.maximum(fxfx**2+fyfy**2-1/wavelength**2,0))
    mask = np.exp(-alpha*dz)


    # Reserving memory for copying the refractive index to a new variable
    # This variable stores the refractive index in the current and previous z plane
    dy = dx # For the sake of readability
    E_Delta_Ln_n = np.zeros((xy_cells,xy_cells,2),dtype=np.complex64)

    # To minimize dynamic memory allocation
    d2Ux_dx2 = np.zeros((xy_cells,xy_cells),dtype=np.complex64)
    d2Ux_dy2 = np.zeros((xy_cells,xy_cells),dtype=np.complex64)
    d2Uy_dx2 = np.zeros((xy_cells,xy_cells),dtype=np.complex64)
    d2Uy_dy2 = np.zeros((xy_cells,xy_cells),dtype=np.complex64)
    d2Uz_dx2 = np.zeros((xy_cells,xy_cells),dtype=np.complex64)
    d2Uz_dy2 = np.zeros((xy_cells,xy_cells),dtype=np.complex64)

    for i in range(1,1+steps):        

        index_L0 = index[:,:,(i-1)%unique_layers]   # I believe these are just views, so no expensive memory allocation happens.
        index_L1 = index[:,:,i%unique_layers]
        index_L2 = index[:,:,(i+1)%unique_layers]

        E_Delta_Ln_n[:,:,1] = (Ux[:,:,1]*(roll(index_L1,-1,axis=1)-index_L1)/dx + Uy[:,:,1]*(roll(index_L1,-1,axis=0)-index_L1)/dy + Uz[:,:,1]*(index_L2-index_L1)/dz)/index_L1
        E_Delta_Ln_n[:,:,0] = (Ux[:,:,0]*(roll(index_L0,-1,axis=1)-index_L0)/dx + Uy[:,:,0]*(roll(index_L0,-1,axis=0)-index_L0)/dy + Uz[:,:,0]*(index_L1-index_L0)/dz)/index_L0

        d2Ux_dx2 = (roll(Ux[:,:,1],1,axis=1)+roll(Ux[:,:,1],-1,axis=1)-2*Ux[:,:,1])
        d2Ux_dy2 = (roll(Ux[:,:,1],1,axis=0)+roll(Ux[:,:,1],-1,axis=0)-2*Ux[:,:,1])
        d2Uy_dx2 = (roll(Uy[:,:,1],1,axis=1)+roll(Uy[:,:,1],-1,axis=1)-2*Uy[:,:,1])
        d2Uy_dy2 = (roll(Uy[:,:,1],1,axis=0)+roll(Uy[:,:,1],-1,axis=0)-2*Uy[:,:,1])
        d2Uz_dx2 = (roll(Uz[:,:,1],1,axis=1)+roll(Uz[:,:,1],-1,axis=1)-2*Uz[:,:,1])
        d2Uz_dy2 = (roll(Uz[:,:,1],1,axis=0)+roll(Uz[:,:,1],-1,axis=0)-2*Uz[:,:,1])

        Ux[:,:,2] = absorption_profile*(2*Ux[:,:,1]-Ux[:,:,0]-(dz_dx**2)*(d2Ux_dx2+d2Ux_dy2)-(dz*k0*index_L1)**2*Ux[:,:,1] - 2*(dz**2/dx)*(roll(E_Delta_Ln_n[:,:,1],-1,axis=1)-E_Delta_Ln_n[:,:,1]))
        Uy[:,:,2] = absorption_profile*(2*Uy[:,:,1]-Uy[:,:,0]-(dz_dx**2)*(d2Uy_dx2+d2Uy_dy2)-(dz*k0*index_L1)**2*Uy[:,:,1] - 2*(dz**2/dy)*(roll(E_Delta_Ln_n[:,:,1],-1,axis=0)-E_Delta_Ln_n[:,:,1]))
        Uz[:,:,2] = absorption_profile*(2*Uz[:,:,1]-Uz[:,:,0]-(dz_dx**2)*(d2Uz_dx2+d2Uz_dy2)-(dz*k0*index_L1)**2*Uz[:,:,1] - 2*dz*(E_Delta_Ln_n[:,:,1]-E_Delta_Ln_n[:,:,0]))
        
        if suppress_evanescent:
            Ux[:,:,2] =  iFFT2(mask*FFT2(Ux[:,:,2]))
            Uy[:,:,2] =  iFFT2(mask*FFT2(Uy[:,:,2]))
            Uz[:,:,2] =  iFFT2(mask*FFT2(Uz[:,:,2]))

        Ux[:,:,0] = Ux[:,:,1]
        Ux[:,:,1] = Ux[:,:,2]
        Uy[:,:,0] = Uy[:,:,1]
        Uy[:,:,1] = Uy[:,:,2]
        Uz[:,:,0] = Uz[:,:,1]
        Uz[:,:,1] = Uz[:,:,2]    
    
    
    
    return Ux, Uy, Uz