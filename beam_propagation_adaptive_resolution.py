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


beam_radius = 1e-3
focus_depth = 2.5e-3    # Depth at which the beam is focused. Note that this is not the focal length in air.
depths = np.array([40e-6,35e-6,30e-6,25e-6,20e-6,15e-6,10e-6,5e-6])      # Calculate the contrast at these tissue depths
section_depth = 10e-6   # Increase resolution every 10 microns in depth
min_FDFD_dx = 80e-9

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
    
    seed_dx = 5*beam_radius/(xy_cells)                 # Resolution for initial seed beam generation only.
    # Propagate depth modulo section_depth first
    spot_size_at_end_of_FDFD_volume = SpotSizeCalculator(focus_depth,beam_radius,n_h,wavelength,(depth-depth%section_depth))
    spot_size_at_start_of_FDFD_volume = SpotSizeCalculator(focus_depth,beam_radius,n_h,wavelength,depth)
    FDFD_dx = min(spot_size_at_end_of_FDFD_volume/30,min_FDFD_dx)
    FDFD_dz = 0.8*FDFD_dx
    propagated_distance = (depth%section_depth)

    print("Propagating "+str(int(10**6*propagated_distance))+" microns with dx = "+str(int(10**9*FDFD_dx))+" nm")

    xy_cells = optimal_cell_size(spot_size_at_start_of_FDFD_volume, FDFD_dx, min_xy_cells)
    
    seed_x = LG_OAM_beam(xy_cells, seed_dx, beam_radius, 1)
    seed_y = 1j*seed_x
    Ex,Ey,Ez,_ = TightFocus(seed_x,seed_y,seed_dx,wavelength,n_h,focus_depth,depth,FDFD_dx,2048)
    Ex2,Ey2,Ez2,_ = TightFocus(seed_x,seed_y,seed_dx,wavelength,n_h,focus_depth,depth-FDFD_dz,FDFD_dx,2048)

    Uz = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
    Uy = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
    Ux = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)

    [Ux[:,:,0], Uy[:,:,0], Uz[:,:,0]] = [Ex, Ey, Ez]
    [Ux[:,:,1], Uy[:,:,1], Uz[:,:,1]] = [Ex2, Ey2, Ez2]

    n = RandomTissue([xy_cells, wavelength, FDFD_dx, FDFD_dz, n_h, ls, g, unique_layers, 0])

    if FDFD_dx > wavelength/((n_h+0.1)*1.41):     # If resolution > lambda/sqrt(2), Evanescent fields blow up. 0.1 adds a small margin of error.
        suppress_evanescent = False
    else:
        suppress_evanescent = True

    Ux,Uy,Uz = Vector_FiniteDifference(Ux,Uy,Uz,section_depth, FDFD_dx, FDFD_dz, xy_cells, n, wavelength, suppress_evanescent)

    distance_left_to_go = (depth-depth%section_depth)
    
    while propagated_distance<depth:



