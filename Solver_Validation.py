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


FDFD_dz = 25e-9         # step size in the direction of beam propagation. Keep this at about lambda/20
wavelength = 750e-9
ls = 59e-6  # Mean free path in tissue
g = 0.90    # Anisotropy factor

# These three parameters determine the NA of the system
beam_radius = 0.8e-3
focus_depth = 2.5e-3    # Depth at which the beam is focused. Note that this is the focal length in the medium.
n_h = 1.33  # Homogenous part of tissue refractive index, also the index of the immersion medium

num_processes = 4       # Number of threads. Make sure there is sufficient RAM
# TODO: Add check to make sure there is sufficient RAM

num_runs = 5           # Simulate each depth this many times
depths = np.array([5e-6,10e-6,15e-6,20e-6,25e-6,30e-6,35e-6,40e-6,45e-6,50e-6,55e-6,60e-6,65e-6,70e-6,75e-6,80e-6])
#depths = np.array([10e-6])
objective_lens_radius_over_beam_radius=2.5  # Self-explanatory: this determines if the objective is overfilled. 2.5 is just about filled.
objective_lens_radius = objective_lens_radius_over_beam_radius*beam_radius

## Change the parameters below if you're sure of what you're doing!
section_depth = 5e-6   # Increase resolution every 5 microns in depth
max_FDFD_dx = 50e-9
suppress_evanescent = True  # Applies the freq. domain filter explained in supplementary material
resolution_factor = 20  # The resolution is increased as the spot size becomes smaller. Increase this for finer resolution. Default = 30
# TODO: Add a setting for lower bound for lateral resolution

unique_layers = 110
# Procedurally generate these many unique layers of tissue. 
# If unique_layers*FDFD_dz is less than the section_depth, the layers are re-used cyclically.
# TODO: Add a safety check to make sure unique_layers*FDFD_dz is at least 1/3rd of section_depth

min_xy_cells = 255      # Minimum cells to be used. This is important because about 10 cells on the outer edge of the simulation form an absorbing boundary.

exportResolution = 1000     # Save fields as a resultSaveResolution x resultSaveResolution matrix. Keep this even.

dx_for_export = 6*SpotSizeCalculator(focus_depth,beam_radius,n_h,wavelength,0)/exportResolution
parameters = Parameters_class(depths,dx_for_export,wavelength,max_FDFD_dx,resolution_factor,FDFD_dz,beam_radius,focus_depth,unique_layers,n_h,ls,g)

def LG_donut_PSF(args):
    depth = args[0]
    run_number = args[1]
    
    print('Simulation (LG) with depth %2.0f um, run number %2.0f starting...' %(10**6*depth, run_number))
    
    # Figure out the lateral discretization of space, i.e., dx
    spot_size = SpotSizeCalculator(focus_depth,beam_radius,n_h,wavelength,depth)
    FDFD_dx = min(spot_size/resolution_factor,max_FDFD_dx)
    xy_cells = optimal_cell_size(spot_size, FDFD_dx, min_xy_cells)
    seed_dx = 5*beam_radius/(xy_cells)

    # Seed with circular polarized LG beam
    seed_x = LG_OAM_beam(xy_cells, seed_dx, beam_radius, 1)
    seed_y = 1j*seed_x

    # Use Debye-Wolf integral to calculate field at two planes near the surface of the tissue.
    Ex,Ey,Ez,_ = TightFocus(seed_x,seed_y,seed_dx,wavelength,n_h,focus_depth,depth,FDFD_dx,3000)
    Ex2,Ey2,Ez2,_ = TightFocus(seed_x,seed_y,seed_dx,wavelength,n_h,focus_depth,depth-FDFD_dz,FDFD_dx,3000)

    Uz = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
    Uy = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
    Ux = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)

    [Ux[:,:,0], Uy[:,:,0], Uz[:,:,0]] = [Ex, Ey, Ez]
    [Ux[:,:,1], Uy[:,:,1], Uz[:,:,1]] = [Ex2, Ey2, Ez2]
    
    # Use the finite difference solver to propagate fields to the focal plane.
    # The Propagate_adaptiveResolution function avoids simulating tissue without any light as detailed in the supplementary material.
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
    
    # Follows the same structure as LG_donut_PSF()
    # HG10 and HG10 beams are separately simulated and the intensities are added, since they are mutually incoherent.
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
    Ex,Ey,Ez,_ = TightFocus(seed_x,seed_y,seed_dx,wavelength,n_h,focus_depth,depth,FDFD_dx,3000)
    Ex2,Ey2,Ez2,_ = TightFocus(seed_x,seed_y,seed_dx,wavelength,n_h,focus_depth,depth-FDFD_dz,FDFD_dx,3000)

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
    print('NA of objective lens is '+str(n_h*objective_lens_radius/focus_depth))
    p = Pool(num_processes)                # This executes everything outside this if statement!

    LG_result = []              # Results will be stored in a list, with each item being an object of class 'Results'
    HG_result = []

    run_number = 0      # Variable to keep count of how many randomized simulations are run
    random_seed = 0     # Run n should be seeded with seed = n, for reproducability.

    # For each simulation, append an entry to the args list containing simulation arguments.
    args = []
    for run_number in range(num_runs):
        for depth in depths:
            args.append([depth, run_number])

    # p.map() runs simulations across multiple processes to speed up the simulation.
    unrolled_results_LG = p.map(LG_donut_PSF, args)
    #unrolled_results_HG = p.map(HG_donut_PSF, args)               # We need to roll up the results into lists later, with each list containing the contrast for all depths for a given instance of tissue.

    # Below, we roll up the results from the pool into a matrix that makes logical sense
    # The LG_results list contains 'num_runs' items. Each item contains the results from all depths specified in the 'depths' list

    tmp_index = 0
    for run_number in range(num_runs):
        tmp_field_exports_LG  = []
        #tmp_field_exports_HG  = []
        for depth in depths:
            tmp_field_exports_LG.append(unrolled_results_LG[tmp_index])     # This only works because of the way args is ordered.
            #tmp_field_exports_HG.append(unrolled_results_HG[tmp_index])
            tmp_index = tmp_index + 1

        # Save results. The Results object is mutable in Python, it needs to be deep-copied
        LG_result.append(copy.deepcopy(Results_class(parameters,tmp_field_exports_LG)))
        #HG_result.append(copy.deepcopy(Results_class(parameters,tmp_field_exports_HG)))
    
    np.save('Results/LG_result', LG_result)
    #np.save('Results/HG_result', HG_result)

    td = timedelta(seconds=time.time() - start_time)
    print('Time taken (hh:mm:ss):', td)
