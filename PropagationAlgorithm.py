#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from FriendlyFourierTransform import FFT2, iFFT2
import matplotlib.pyplot as plt
from GenerateRandomTissue import RandomTissue
from FriendlyFourierTransform import optimal_cell_size
import copy
from scipy.interpolate import RegularGridInterpolator
from helper_classes import Parameters_class,Results_class
from DebyeWolfIntegral import SpotSizeCalculator

def Propagate_adaptiveResolution(Ux, Uy, Uz, distance, dx, dz, run_number, focus_depth, beam_radius, n_h, wavelength, ls, g, max_FDFD_dx = 50e-9, resolution_factor = 20, min_xy_cells = 250, section_depth = 5e-6, suppress_evanescent = True, min_index_layers = 110):

    if np.abs(distance/section_depth-np.rint(distance/section_depth))>0.01:
        print('Warning: Rounding tissue thickness to a multiple of the adaptive thickness parameter.')

    num_steps = int(np.rint(distance/section_depth))
    FDFD_dx = dx
    FDFD_dz = dz
    xy_cells =np.shape(Ux)[0]
    unique_layers = int(min(min_index_layers,section_depth/dz))

    for step_num in range(num_steps):

        remaining_distance = distance - step_num * section_depth
        spot_size_at_start_of_FDFD_volume = SpotSizeCalculator(focus_depth,beam_radius,n_h,wavelength,remaining_distance)
        spot_size_at_end_of_FDFD_volume = SpotSizeCalculator(focus_depth,beam_radius,n_h,wavelength,remaining_distance-section_depth)

        original_axis = FDFD_dx*np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
        FDFD_dx = min(spot_size_at_end_of_FDFD_volume/resolution_factor,max_FDFD_dx)
        xy_cells = optimal_cell_size(spot_size_at_start_of_FDFD_volume, FDFD_dx, min_xy_cells)
        new_axes = FDFD_dx*np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
        xx_new, yy_new = np.meshgrid(new_axes,new_axes, indexing='ij')  # Mesh grid used for exporting data

        Uz_new = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
        Uy_new = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
        Ux_new = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)

        # Bounds error has to be set to False in case (depth%section_depth) is zero.
        Ux_new[:,:,0] = RegularGridInterpolator((original_axis,original_axis),Ux[:,:,0], bounds_error = False, fill_value = 0, method='linear')((xx_new, yy_new))
        Ux_new[:,:,1] = RegularGridInterpolator((original_axis,original_axis),Ux[:,:,1], bounds_error = False, fill_value = 0, method='linear')((xx_new, yy_new))
        Uy_new[:,:,0] = RegularGridInterpolator((original_axis,original_axis),Uy[:,:,0], bounds_error = False, fill_value = 0, method='linear')((xx_new, yy_new))
        Uy_new[:,:,1] = RegularGridInterpolator((original_axis,original_axis),Uy[:,:,1], bounds_error = False, fill_value = 0, method='linear')((xx_new, yy_new))
        Uz_new[:,:,0] = RegularGridInterpolator((original_axis,original_axis),Uz[:,:,0], bounds_error = False, fill_value = 0, method='linear')((xx_new, yy_new))
        Uz_new[:,:,1] = RegularGridInterpolator((original_axis,original_axis),Uz[:,:,1], bounds_error = False, fill_value = 0, method='linear')((xx_new, yy_new))

        [Ux, Uy, Uz] = [Ux_new, Uy_new, Uz_new]
        n = RandomTissue([xy_cells, wavelength, FDFD_dx, FDFD_dz, n_h, ls, g, unique_layers, run_number])
        #n = n_h*np.ones((xy_cells,xy_cells,20))
        Ux,Uy,Uz = Propagate(Ux,Uy,Uz,section_depth, FDFD_dx, FDFD_dz, xy_cells, n, wavelength, suppress_evanescent)

    return Ux,Uy,Uz,FDFD_dx,xy_cells

def Propagate(Ux, Uy, Uz, distance, dx, dz, xy_cells, index, wavelength, suppress_evanescent = True):
    
    # This has the same problem: If dx < lambda/sqrt(2), the field blows up.
    # Implementation follows Eq. 3-8 in Goodman Fourier Optics

    # x axis is the second index (axis=1). y axis is first (axis=0)
    # This makes sure pcolor represents fields accurately
    InitialPower = np.sum((np.abs(Ux[:,:,0])**2+np.abs(Uy[:,:,0])**2+np.abs(Uz[:,:,0])**2)*dx**2)
    if xy_cells%2 == 1:
        ValueError('Cell size has to be even')

    indices = np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
    f = 1/(dx*xy_cells)*indices
    k0 = 2*np.pi/wavelength
    steps = int(distance/dz)
    fxfx,fyfy = np.meshgrid(f,f)
    
    
    # Absorption boundary condition profile = complex part of index
    xy_range = dx*xy_cells/2
    xx, yy = np.meshgrid(dx*indices,dx*indices)
    absorption_profile = np.exp(-(np.exp(0.1*(np.abs(xx)/dx-xy_cells/2+10)) + np.exp(0.1*(np.abs(yy)/dx-xy_cells/2+10))))
    # Cannot use imaginary index because it makes the equation numerically unstable. If necessary, apply this profile after U[:,:,2] is calculated as
    # U[:,:,2] = exp(-alpha*dz)*U[:,:,2]
    unique_layers = np.shape(index)[2]
    dz_dx = dz/dx
    
    # Masking evanescent fields
    f = 1/(dx*xy_cells)*indices
    fxfx,fyfy = np.meshgrid(f,f)
    #mask = ((fxfx**2+fyfy**2)<(1/wavelength)**2).astype(float)

    ## Type 2 Fourier mask
    # Explanation: We attenuate evanescent fields that meet the condition fx^2 + fy^2 > n/lambda
    # Since n is not homogenous, we take the 10 percentile low value of n in the whole volume.
    alpha = 2*np.pi*np.sqrt(np.maximum(fxfx**2+fyfy**2-np.percentile(index,10)**2/wavelength**2,0))
    mask = np.exp(-alpha*dz)

    # Set the very high frequency terms to 0 in the mask
    # mask[np.where((fxfx**2+fyfy**2)>(4*np.average(index)/wavelength)**2)] = 0

    # Reserving memory for copying the refractive index to a new variable
    # This variable stores the refractive index in the current and previous z plane
    dy = dx # For the sake of readability
    E_Delta_Ln_n = np.zeros((xy_cells,xy_cells,2),dtype=np.complex64)

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

        E_Delta_Ln_n[:,:,1] = (Ux[:,:,1]*(np.roll(index_L1,-1,axis=1)-index_L1)/dx + Uy[:,:,1]*(np.roll(index_L1,-1,axis=0)-index_L1)/dy + Uz[:,:,1]*(index_L2-index_L1)/dz)/index_L1
        E_Delta_Ln_n[:,:,0] = (Ux[:,:,0]*(np.roll(index_L0,-1,axis=1)-index_L0)/dx + Uy[:,:,0]*(np.roll(index_L0,-1,axis=0)-index_L0)/dy + Uz[:,:,0]*(index_L1-index_L0)/dz)/index_L0

        d2Ux_dx2 = (np.roll(Ux[:,:,1],1,axis=1)+np.roll(Ux[:,:,1],-1,axis=1)-2*Ux[:,:,1])
        d2Ux_dy2 = (np.roll(Ux[:,:,1],1,axis=0)+np.roll(Ux[:,:,1],-1,axis=0)-2*Ux[:,:,1])
        d2Uy_dx2 = (np.roll(Uy[:,:,1],1,axis=1)+np.roll(Uy[:,:,1],-1,axis=1)-2*Uy[:,:,1])
        d2Uy_dy2 = (np.roll(Uy[:,:,1],1,axis=0)+np.roll(Uy[:,:,1],-1,axis=0)-2*Uy[:,:,1])
        d2Uz_dx2 = (np.roll(Uz[:,:,1],1,axis=1)+np.roll(Uz[:,:,1],-1,axis=1)-2*Uz[:,:,1])
        d2Uz_dy2 = (np.roll(Uz[:,:,1],1,axis=0)+np.roll(Uz[:,:,1],-1,axis=0)-2*Uz[:,:,1])

        Ux[:,:,2] = absorption_profile*(2*Ux[:,:,1]-Ux[:,:,0]-(dz_dx**2)*(d2Ux_dx2+d2Ux_dy2)-(dz*k0*index_L1)**2*Ux[:,:,1] - 2*(dz**2/dx)*(np.roll(E_Delta_Ln_n[:,:,1],-1,axis=1)-E_Delta_Ln_n[:,:,1]))
        Uy[:,:,2] = absorption_profile*(2*Uy[:,:,1]-Uy[:,:,0]-(dz_dx**2)*(d2Uy_dx2+d2Uy_dy2)-(dz*k0*index_L1)**2*Uy[:,:,1] - 2*(dz**2/dy)*(np.roll(E_Delta_Ln_n[:,:,1],-1,axis=0)-E_Delta_Ln_n[:,:,1]))
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
    
    FinalPower = np.sum((np.abs(Ux[:,:,0])**2+np.abs(Uy[:,:,0])**2+np.abs(Uz[:,:,0])**2)*dx**2)
    PowerScalingFactor = np.sqrt(InitialPower/FinalPower)
    Ux = PowerScalingFactor*Ux
    Uy = PowerScalingFactor*Uy
    Uz = PowerScalingFactor*Uz
    
    
    return Ux, Uy, Uz