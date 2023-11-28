import numpy as np
import time
from PropagationAlgorithm import Propagate, Propagate_adaptiveResolution
from SeedBeams import LG_OAM_beam, HG_beam
from FieldPlots import plot_HG, plot_LG
from GenerateRandomTissue import RandomTissue
from DebyeWolfIntegral import TightFocus, SpotSizeCalculator
from FriendlyFourierTransform import optimal_cell_size
import copy
from datetime import timedelta
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from helper_classes import Parameters_class,Results_class
from multiprocessing import Pool


FDFD_dz = 25e-9
beam_radius = 1e-3
focus_depth = 2.5e-3    # Depth at which the beam is focused. Note that this is not the focal length in air.
#depths = np.array([40e-6,35e-6,30e-6,25e-6,20e-6,15e-6,10e-6,5e-6])      # Calculate the contrast at these tissue depths
num_processes = 6
num_runs = 20
depths = np.array([5e-6,10e-6,15e-6,20e-6,25e-6,30e-6,35e-6,40e-6,45e-6,50e-6,55e-6,60e-6])
section_depth = 5e-6   # Increase resolution every 5 microns in depth
max_FDFD_dx = 50e-9
suppress_evanescent = True
resolution_factor = 30  # Increase this for finer resolution
n_h = 1.33  # Homogenous part of refractive index
ls = 59e-6  # Mean free path in tissue
g = 0.90    # Anisotropy factor

unique_layers = 110    # Unique layers of refractive index for procedural generation of tissue. Unclear what's the effect of making this small.
wavelength = 500e-9
min_xy_cells = 255      # Minimum cells to be used. This is important because 10 or so cells on the edge form an absorbing boundary.

exportResolution = 1000     # Save fields as a resultSaveResolution x resultSaveResolution matrix. Keep this even.
dx_for_export = 6*SpotSizeCalculator(focus_depth,beam_radius,n_h,wavelength,0)/exportResolution
parameters = Parameters_class(depths,dx_for_export,wavelength,max_FDFD_dx,resolution_factor,FDFD_dz,beam_radius,focus_depth,unique_layers,n_h,ls,g)

def LG_donut_PSF(args):
    depth = args[0]
    run_number = args[1]
    
    print('Simulation (LG) with depth %2.0f um, run number %2.0f starting...' %(10**6*depth, run_number))

    spot_size = SpotSizeCalculator(focus_depth,beam_radius,n_h,wavelength,depth)
    FDFD_dx = min(spot_size/resolution_factor,max_FDFD_dx)
    xy_cells = optimal_cell_size(spot_size, FDFD_dx, min_xy_cells)
    seed_dx = 5*beam_radius/(xy_cells)
    seed_x = LG_OAM_beam(xy_cells, seed_dx, beam_radius, 1)
    seed_y = 1j*seed_x
    Ex,Ey,Ez,_ = TightFocus(seed_x,seed_y,seed_dx,wavelength,n_h,focus_depth,depth,FDFD_dx,2048)
    Ex2,Ey2,Ez2,_ = TightFocus(seed_x,seed_y,seed_dx,wavelength,n_h,focus_depth,depth-FDFD_dz,FDFD_dx,2048)

    Uz = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
    Uy = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
    Ux = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)

    [Ux[:,:,0], Uy[:,:,0], Uz[:,:,0]] = [Ex, Ey, Ez]
    [Ux[:,:,1], Uy[:,:,1], Uz[:,:,1]] = [Ex2, Ey2, Ez2]
    
    Ux,Uy,Uz,FDFD_dx,xy_cells = Propagate_adaptiveResolution(Ux, Uy, Uz, depth, FDFD_dx, FDFD_dz, run_number, focus_depth, beam_radius, n_h, wavelength, ls, g, max_FDFD_dx = max_FDFD_dx, resolution_factor = resolution_factor, min_xy_cells = min_xy_cells, section_depth = section_depth, suppress_evanescent = True, min_index_layers = unique_layers)
    
    LG_Focus_Intensity = np.abs(Ux[:,:,0])**2+np.abs(Uy[:,:,0])**2+np.abs(Uz[:,:,0])**2
    original_axis = 10**6*FDFD_dx*np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
    export_axes = 10**6*dx_for_export*np.linspace(-exportResolution/2,exportResolution/2-1,exportResolution,dtype=np.int_)
    xx_export, yy_export = np.meshgrid(export_axes,export_axes, indexing='ij')  # Mesh grid used for exporting data
    export_field = RegularGridInterpolator((original_axis,original_axis),LG_Focus_Intensity, bounds_error = True, method='linear')((xx_export, yy_export))

    plot_LG(focus_depth,beam_radius,n_h,wavelength,xy_cells,FDFD_dx,LG_Focus_Intensity,depth,run_number)
    print('Simulation (LG) with depth %2.0f um, run number %2.0f exiting' %(10**6*depth, run_number))
    return export_field

def HG_donut_PSF(args):
    depth = args[0]
    run_number = args[1]
    
    print('Simulation (HG) with depth %2.0f um, run number %2.0f starting...' %(10**6*depth, run_number))

    ## HG01
    spot_size = SpotSizeCalculator(focus_depth,beam_radius,n_h,wavelength,depth)
    FDFD_dx = min(spot_size/resolution_factor,max_FDFD_dx)
    xy_cells = optimal_cell_size(spot_size, FDFD_dx, min_xy_cells)
    seed_dx = 5*beam_radius/(xy_cells)
    (u,v) = (1,0)   # Mode numbers for HG beam
    seed_y = HG_beam(xy_cells, seed_dx, beam_radius, u,v)    # This is well behaved and does not fill in at the focus
    seed_x = np.zeros(seed_y.shape)
    Ex,Ey,Ez,_ = TightFocus(seed_x,seed_y,seed_dx,wavelength,n_h,focus_depth,depth,FDFD_dx,2048)
    Ex2,Ey2,Ez2,_ = TightFocus(seed_x,seed_y,seed_dx,wavelength,n_h,focus_depth,depth-FDFD_dz,FDFD_dx,2048)
    Uz = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
    Uy = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
    Ux = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)

    [Ux[:,:,0], Uy[:,:,0], Uz[:,:,0]] = [Ex, Ey, Ez]
    [Ux[:,:,1], Uy[:,:,1], Uz[:,:,1]] = [Ex2, Ey2, Ez2]
    
    Ux,Uy,Uz,_,_ = Propagate_adaptiveResolution(Ux, Uy, Uz, depth, FDFD_dx, FDFD_dz, run_number, focus_depth, beam_radius, n_h, wavelength, ls, g, max_FDFD_dx = max_FDFD_dx, resolution_factor = resolution_factor, min_xy_cells = min_xy_cells, section_depth = section_depth, suppress_evanescent = suppress_evanescent, min_index_layers = unique_layers)
    
    HG10_Focus_Intensity = np.abs(Ux[:,:,0])**2+np.abs(Uy[:,:,0])**2+np.abs(Uz[:,:,0])**2   # HG10 will also have the same FDFD_dx so we don't need to remember it.

    ## HG10
    (u,v) = (0,1)   # Mode numbers for HG beam
    seed_x = HG_beam(xy_cells, seed_dx, beam_radius, u,v)    # Note that seed_y is first now. This is well behaved and does not fill in at the focus
    seed_y = np.zeros(seed_x.shape)
    Ex,Ey,Ez,_ = TightFocus(seed_x,seed_y,seed_dx,wavelength,n_h,focus_depth,depth,FDFD_dx,2048)
    Ex2,Ey2,Ez2,_ = TightFocus(seed_x,seed_y,seed_dx,wavelength,n_h,focus_depth,depth-FDFD_dz,FDFD_dx,2048)

    Uz = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
    Uy = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
    Ux = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
    [Ux[:,:,0], Uy[:,:,0], Uz[:,:,0]] = [Ex, Ey, Ez]
    [Ux[:,:,1], Uy[:,:,1], Uz[:,:,1]] = [Ex2, Ey2, Ez2]
    
    Ux,Uy,Uz,FDFD_dx,xy_cells = Propagate_adaptiveResolution(Ux, Uy, Uz, depth, FDFD_dx, FDFD_dz, run_number, focus_depth, beam_radius, n_h, wavelength, ls, g, max_FDFD_dx = max_FDFD_dx, resolution_factor = resolution_factor, min_xy_cells = min_xy_cells, section_depth = section_depth, suppress_evanescent = suppress_evanescent, min_index_layers = unique_layers)
    
    HG01_Focus_Intensity = np.abs(Ux[:,:,0])**2+np.abs(Uy[:,:,0])**2+np.abs(Uz[:,:,0])**2   # HG10 will also have the same FDFD_dx so we don't need to remember it.

    HG_Focus_Intensity = (HG01_Focus_Intensity+HG10_Focus_Intensity)
    plot_HG(focus_depth,beam_radius,n_h,wavelength,xy_cells,FDFD_dx,HG10_Focus_Intensity,HG01_Focus_Intensity,HG_Focus_Intensity,depth,run_number)
    
    original_axis = 10**6*FDFD_dx*np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
    export_axes = 10**6*dx_for_export*np.linspace(-exportResolution/2,exportResolution/2-1,exportResolution,dtype=np.int_)
    xx_export, yy_export = np.meshgrid(export_axes,export_axes, indexing='ij')  # Mesh grid used for exporting data
    export_field = RegularGridInterpolator((original_axis,original_axis),HG_Focus_Intensity, bounds_error = True, method='linear')((xx_export, yy_export))

    print('Simulation (HG) with depth %2.0f um, run number %2.0f exiting' %(10**6*depth, run_number))
    return export_field



if __name__ == '__main__':
    start_time = time.time()
    print('NA of objective lens is '+str(n_h*beam_radius*1.5/focus_depth))
    p = Pool(num_processes)                # Remember! This executes everything outside this if statement!

    LG_result = []              # List of objects of class 'Results'
    HG_result = []

    run_number = 0
    random_seed = 0     # Run n should be seeded with seed = n, for reproducability.
    args = []
    for run_number in range(num_runs):
        for depth in depths:
            args.append([depth, run_number])

    unrolled_results_HG = p.map(LG_donut_PSF, args)
    unrolled_results_LG = p.map(HG_donut_PSF, args)               # We need to roll up the results into lists, with each list containing the contrast for all depths for a given instance of tissue.

    tmp_index = 0
    for run_number in range(num_runs):
        tmp_field_exports_LG  = []
        tmp_field_exports_HG  = []
        for depth in depths:
            tmp_field_exports_LG.append(unrolled_results_LG[tmp_index])     # This only works because of the way args is ordered.
            tmp_field_exports_HG.append(unrolled_results_HG[tmp_index])
            tmp_index = tmp_index + 1

        # Save results. The Results object is mutable in Python, so I need to deepcopy it.
        LG_result.append(copy.deepcopy(Results_class(parameters,tmp_field_exports_LG)))
        HG_result.append(copy.deepcopy(Results_class(parameters,tmp_field_exports_HG)))
    
    np.save('Results/Contrast_LG', LG_result)
    np.save('Results/Contrast_HG', HG_result)

    td = timedelta(seconds=time.time() - start_time)
    print('Time taken (hh:mm:ss):', td)
