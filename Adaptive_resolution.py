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
from ParallelTightFocusContrastVsDepth import Results
from multiprocessing import Pool


FDFD_dz = 25e-9
beam_radius = 1e-3
focus_depth = 2.5e-3    # Depth at which the beam is focused. Note that this is not the focal length in air.
#depths = np.array([40e-6,35e-6,30e-6,25e-6,20e-6,15e-6,10e-6,5e-6])      # Calculate the contrast at these tissue depths
num_processes = 3
num_runs = 5
depths = np.array([5e-6,10e-6,15e-6,20e-6,25e-6,30e-6,35e-6])
section_depth = 10e-6   # Increase resolution every 10 microns in depth
max_FDFD_dx = 40e-9
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

LG_result = []
tmp_field_exports_LG  = [np.zeros((exportResolution,exportResolution))]*len(depths)

def Tightfocus_LG_adaptive(args):
    depth = args[0]
    shared_mem_name = args[1]
    run_number = args[2]
    print('Simulation (LG) with depth %2.0f um, run number %2.0f starting...' %(10**6*depth, run_number))

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
    #axis = FDFD_dx*np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
    
    if propagated_distance>FDFD_dz:
        n = RandomTissue([xy_cells, wavelength, FDFD_dx, FDFD_dz, n_h, ls, g, unique_layers, run_number])
        #n = n_h*np.ones((xy_cells,xy_cells,20))
        Ux,Uy,Uz = Vector_FiniteDifference(Ux,Uy,Uz,propagated_distance, FDFD_dx, FDFD_dz, xy_cells, n, wavelength, suppress_evanescent)
    distance_left_to_go = (depth-propagated_distance)
    
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

        # Bounds error has to be set to False in case (depth%section_depth) is zero.
        Ux_new[:,:,0] = RegularGridInterpolator((original_axis,original_axis),Ux[:,:,0], bounds_error = False, method='linear')((xx_new, yy_new))
        Ux_new[:,:,1] = RegularGridInterpolator((original_axis,original_axis),Ux[:,:,1], bounds_error = False, method='linear')((xx_new, yy_new))
        Uy_new[:,:,0] = RegularGridInterpolator((original_axis,original_axis),Uy[:,:,0], bounds_error = False, method='linear')((xx_new, yy_new))
        Uy_new[:,:,1] = RegularGridInterpolator((original_axis,original_axis),Uy[:,:,1], bounds_error = False, method='linear')((xx_new, yy_new))
        Uz_new[:,:,0] = RegularGridInterpolator((original_axis,original_axis),Uz[:,:,0], bounds_error = False, method='linear')((xx_new, yy_new))
        Uz_new[:,:,1] = RegularGridInterpolator((original_axis,original_axis),Uz[:,:,1], bounds_error = False, method='linear')((xx_new, yy_new))

        [Ux, Uy, Uz] = [Ux_new, Uy_new, Uz_new]
        n = RandomTissue([xy_cells, wavelength, FDFD_dx, FDFD_dz, n_h, ls, g, unique_layers, run_number])
        #n = n_h*np.ones((xy_cells,xy_cells,20))
        Ux,Uy,Uz = Vector_FiniteDifference(Ux,Uy,Uz,section_depth, FDFD_dx, FDFD_dz, xy_cells, n, wavelength, suppress_evanescent)
        distance_left_to_go = distance_left_to_go - section_depth

    LG_Focus_Intensity = np.abs(Ux[:,:,0])**2+np.abs(Uy[:,:,0])**2+np.abs(Uz[:,:,0])**2
    #exportResolution = 1000     # Save fields as a resultSaveResolution x resultSaveResolution matrix. Keep this even.
    #dx_for_export = 6*SpotSizeCalculator(focus_depth,beam_radius,n_h,wavelength,0)/exportResolution
    original_axis = 10**6*FDFD_dx*np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
    export_axes = 10**6*dx_for_export*np.linspace(-exportResolution/2,exportResolution/2-1,exportResolution,dtype=np.int_)
    xx_export, yy_export = np.meshgrid(export_axes,export_axes, indexing='ij')  # Mesh grid used for exporting data
    export_field = RegularGridInterpolator((original_axis,original_axis),LG_Focus_Intensity, bounds_error = True, method='linear')((xx_export, yy_export))

    plt.figure()
    plt.pcolormesh(xx_export, yy_export,export_field)
    plt.colorbar()
    plt.title(str("{:02d}".format(int(1e6*depth)))+'um_run'+str("{:02d}".format(run_number)))
    plt.savefig('Results/LG_'+str("{:02d}".format(int(1e6*depth)))+'um_run'+str("{:02d}".format(run_number))+'.png', bbox_inches = 'tight', dpi=500)
    plt.close()
    print('Simulation (LG) with depth %2.0f um, run number %2.0f exiting' %(10**6*depth, run_number))
    return 0, 0, export_field

if __name__ == '__main__':
    start_time = time.time()
    print('NA of objective lens is '+str(n_h*beam_radius*1.5/focus_depth))
    p = Pool(num_processes)                # Remember! This executes everything outside this if statement!

    LG_result = []              # List of objects of class 'Results'
    
    run_number = 0
    random_seed = 0     # Run n should be seeded with seed = n, for reproducability.
    args = []
    for run_number in range(num_runs):
        for depth in depths:
            args.append([depth,'', run_number])

    unrolled_results_LG = p.map(Tightfocus_LG_adaptive, args)               # We need to roll up the results into lists, with each list containing the contrast for all depths for a given instance of tissue.

    tmp_index = 0
    for run_number in range(num_runs):
        tmp_field_exports_LG  = []
        for depth in depths:
            tmp_field_exports_LG.append(unrolled_results_LG[tmp_index])     # This only works because of the way args is ordered.
            tmp_index = tmp_index + 1

        # Save results. The Results object is mutable in Python, so I need to deepcopy it.
        LG_result.append(copy.deepcopy(Results(np.zeros(len(depths)),np.zeros(len(depths)),tmp_field_exports_LG)))
    
    np.save('Results/Contrast_LG', LG_result)

    td = timedelta(seconds=time.time() - start_time)
    print('Time taken (hh:mm:ss):', td)