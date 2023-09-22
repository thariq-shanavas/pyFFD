import numpy as np
import time
from PropagationAlgorithm import Vector_FiniteDifference
from SeedBeams import LG_OAM_beam, HG_beam
from FieldPlots import VortexNull, plot_HG, plot_LG
from GenerateRandomTissue import RandomTissue
from DebyeWolfIntegral import TightFocus, SpotSizeCalculator
from FriendlyFourierTransform import optimal_cell_size
import copy
from datetime import timedelta
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt


FDFD_dz = 25e-9
beam_radius = 1e-3
focus_depth = 2.5e-3    # Depth at which the beam is focused. Note that this is not the focal length in air.
#depths = np.array([40e-6,35e-6,30e-6,25e-6,20e-6,15e-6,10e-6,5e-6])      # Calculate the contrast at these tissue depths
depths = np.array([15e-6])
section_depth = 10e-6   # Increase resolution every 10 microns in depth
max_FDFD_dx = 50e-9
suppress_evanescent = True
resolution_factor = 40  # Increase this for finer resolution

n_h = 1.33  # Homogenous part of refractive index
ls = 15e-6  # Mean free path in tissue
g = 0.92    # Anisotropy factor

unique_layers = 111    # Unique layers of refractive index for procedural generation of tissue. Unclear what's the effect of making this small.
wavelength = 500e-9
min_xy_cells = 255      # Minimum cells to be used. This is important because 10 or so cells on the edge form an absorbing boundary.

exportResolution = 1000     # Save fields as a resultSaveResolution x resultSaveResolution matrix. Keep this even.
dx_for_export = 6*SpotSizeCalculator(focus_depth,beam_radius,n_h,wavelength,0)/exportResolution

# We share the same procedurally generated tissue across many simulations. We find the volume needed for the largest simulation
for depth in depths:
    
    
    # Propagate depth modulo section_depth first
    spot_size_at_end_of_FDFD_volume = SpotSizeCalculator(focus_depth,beam_radius,n_h,wavelength,(depth-depth%section_depth))
    spot_size_at_start_of_FDFD_volume = SpotSizeCalculator(focus_depth,beam_radius,n_h,wavelength,depth)
    FDFD_dx = min(spot_size_at_end_of_FDFD_volume/resolution_factor,max_FDFD_dx)
    #FDFD_dz = 0.8*FDFD_dx
    propagated_distance = (depth%section_depth)

    xy_cells = optimal_cell_size(spot_size_at_start_of_FDFD_volume, FDFD_dx, min_xy_cells)
    seed_dx = 5*beam_radius/(xy_cells)                 # Resolution for initial seed beam generation only.
    seed_x = LG_OAM_beam(xy_cells, seed_dx, beam_radius, 1)
    seed_y = 1j*seed_x
    Ex,Ey,Ez,_ = TightFocus(seed_x,seed_y,seed_dx,wavelength,n_h,focus_depth,depth,FDFD_dx,2048)
    Ex2,Ey2,Ez2,_ = TightFocus(seed_x,seed_y,seed_dx,wavelength,n_h,focus_depth,depth-FDFD_dz,FDFD_dx,2048)

    Uz = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
    Uy = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
    Ux = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)

    [Ux[:,:,0], Uy[:,:,0], Uz[:,:,0]] = [Ex, Ey, Ez]
    [Ux[:,:,1], Uy[:,:,1], Uz[:,:,1]] = [Ex2, Ey2, Ez2]
    axis = FDFD_dx*np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
    plt.pcolormesh(axis,axis,np.abs(Ux[:,:,0])**2+np.abs(Uy[:,:,0])**2+np.abs(Uz[:,:,0])**2)
    plt.colorbar()
    plt.show()

    n = RandomTissue([xy_cells, wavelength, FDFD_dx, FDFD_dz, n_h, ls, g, unique_layers, 0])
    #n = n_h*np.ones((xy_cells,xy_cells,20))
    print("Propagating "+str(int(10**6*propagated_distance))+" microns with dx = "+str(int(10**9*FDFD_dx))+" nm and xy_cells = "+str(xy_cells))


    Ux,Uy,Uz = Vector_FiniteDifference(Ux,Uy,Uz,propagated_distance, FDFD_dx, FDFD_dz, xy_cells, n, wavelength, suppress_evanescent)
    plt.pcolormesh(axis,axis,np.abs(Ux[:,:,0])**2+np.abs(Uy[:,:,0])**2+np.abs(Uz[:,:,0])**2)
    plt.colorbar()
    plt.show()

    distance_left_to_go = (depth-depth%section_depth)
    
    for _ in range(int(distance_left_to_go/section_depth)):
        spot_size_at_start_of_FDFD_volume = SpotSizeCalculator(focus_depth,beam_radius,n_h,wavelength,distance_left_to_go)
        spot_size_at_end_of_FDFD_volume = SpotSizeCalculator(focus_depth,beam_radius,n_h,wavelength,distance_left_to_go-section_depth)

        original_axis = FDFD_dx*np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
        FDFD_dx = min(spot_size_at_end_of_FDFD_volume/resolution_factor,max_FDFD_dx)
        #FDFD_dz = 0.8*FDFD_dx
        xy_cells = optimal_cell_size(spot_size_at_start_of_FDFD_volume, FDFD_dx, min_xy_cells)
        new_axes = FDFD_dx*np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
        xx_new, yy_new = np.meshgrid(new_axes,new_axes, indexing='ij')  # Mesh grid used for exporting data

        Uz_new = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
        Uy_new = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
        Ux_new = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)

        Ux_new[:,:,0] = RegularGridInterpolator((original_axis,original_axis),Ux[:,:,0], bounds_error = True, method='linear')((xx_new, yy_new))
        Ux_new[:,:,1] = RegularGridInterpolator((original_axis,original_axis),Ux[:,:,1], bounds_error = True, method='linear')((xx_new, yy_new))
        Uy_new[:,:,0] = RegularGridInterpolator((original_axis,original_axis),Uy[:,:,0], bounds_error = True, method='linear')((xx_new, yy_new))
        Uy_new[:,:,1] = RegularGridInterpolator((original_axis,original_axis),Uy[:,:,1], bounds_error = True, method='linear')((xx_new, yy_new))
        Uz_new[:,:,0] = RegularGridInterpolator((original_axis,original_axis),Uz[:,:,0], bounds_error = True, method='linear')((xx_new, yy_new))
        Uz_new[:,:,1] = RegularGridInterpolator((original_axis,original_axis),Uz[:,:,1], bounds_error = True, method='linear')((xx_new, yy_new))

        [Ux, Uy, Uz] = [Ux_new, Uy_new, Uz_new]
        n = RandomTissue([xy_cells, wavelength, FDFD_dx, FDFD_dz, n_h, ls, g, unique_layers, 0])
        #n = n_h*np.ones((xy_cells,xy_cells,20))
        print("Propagating "+str(int(10**6*section_depth))+" microns with dx = "+str(int(10**9*FDFD_dx))+" nm and xy_cells = "+str(xy_cells))
        plt.pcolormesh(xx_new, yy_new, np.abs(Ux[:,:,0])**2+np.abs(Uy[:,:,0])**2+np.abs(Uz[:,:,0])**2)
        plt.colorbar()
        plt.show()
        Ux,Uy,Uz = Vector_FiniteDifference(Ux,Uy,Uz,section_depth, FDFD_dx, FDFD_dz, xy_cells, n, wavelength, suppress_evanescent)
        distance_left_to_go = distance_left_to_go - section_depth

LG_Focus_Intensity = np.abs(Ux[:,:,0])**2+np.abs(Uy[:,:,0])**2+np.abs(Uz[:,:,0])**2
exportResolution = 1000     # Save fields as a resultSaveResolution x resultSaveResolution matrix. Keep this even.
dx_for_export = 6*SpotSizeCalculator(focus_depth,beam_radius,n_h,wavelength,0)/exportResolution
original_axis = 10**6*FDFD_dx*np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
export_axes = 10**6*dx_for_export*np.linspace(-exportResolution/2,exportResolution/2-1,exportResolution,dtype=np.int_)
xx_export, yy_export = np.meshgrid(export_axes,export_axes, indexing='ij')  # Mesh grid used for exporting data
export_field = RegularGridInterpolator((original_axis,original_axis),LG_Focus_Intensity, bounds_error = True, method='linear')((xx_export, yy_export))

plt.pcolormesh(xx_export, yy_export,export_field)
plt.colorbar()
plt.title("FDFD donut")
plt.show()

Ex,Ey,Ez,_ = TightFocus(seed_x,seed_y,seed_dx,wavelength,n_h,focus_depth,0,dx_for_export,4096)
Ideal_PSF = np.abs(Ex)**2+np.abs(Ey)**2+np.abs(Ez)**2
xy_cells = np.shape(Ex)[0]
axis = 10**6*dx_for_export*np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
plt.pcolormesh(axis,axis,Ideal_PSF)
plt.colorbar()
plt.title("Ideal donut")
plt.show()