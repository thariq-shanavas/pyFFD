import numpy as np
import time
from PropagationAlgorithm import Vector_FiniteDifference
from SeedBeams import LG_OAM_beam, HG_beam
from FieldPlots import VortexNull, plot_HG, plot_LG
from GenerateRandomTissue import RandomTissue
from DebyeWolfIntegral import TightFocus, SpotSizeCalculator
from multiprocessing import Pool, shared_memory
from FriendlyFourierTransform import optimal_cell_size
import copy
from datetime import timedelta
from scipy.interpolate import RegularGridInterpolator

# Simulation parameters
beam_radius = 1e-3
focus_depth = 2.5e-3    # Depth at which the beam is focused. Note that this is not the focal length in air.
depths = np.array([40e-6,35e-6,30e-6,25e-6,20e-6,15e-6,10e-6,5e-6])      # Calculate the contrast at these tissue depths
n_h = 1.33  # Homogenous part of refractive index
ls = 15e-6  # Mean free path in tissue
g = 0.92    # Anisotropy factor

FDFD_dx = 50e-9     # Recommended to keep this below 50 nm ideally.
FDFD_dz = FDFD_dx * 0.8
unique_layers = 100    # Unique layers of refractive index for procedural generation of tissue. Unclear what's the effect of making this small.
wavelength = 500e-9
min_xy_cells = 255      # Minimum cells to be used. This is important because 10 or so cells on the edge form an absorbing boundary.

exportResolution = 1000     # Save fields as a resultSaveResolution x resultSaveResolution matrix. Keep this even.
dx_for_export = 6*SpotSizeCalculator(focus_depth,beam_radius,n_h,wavelength,0)/exportResolution

# We share the same procedurally generated tissue across many simulations. We find the volume needed for the largest simulation
max_spot_size_at_start_of_FDFD_volume = SpotSizeCalculator(focus_depth,beam_radius,n_h,wavelength,np.max(depths))
global_xy_cells = optimal_cell_size(max_spot_size_at_start_of_FDFD_volume, FDFD_dx, min_xy_cells)

if FDFD_dz>FDFD_dx or FDFD_dz>wavelength/10:
    raise ValueError('Reduce FDFD_dz!')

if FDFD_dx > wavelength/((n_h+0.1)*1.41):     # If resolution > lambda/sqrt(2), Evanescent fields blow up. 0.1 adds a small margin of error.
    suppress_evanescent = False
else:
    suppress_evanescent = True
 
class Results:
    # Saves results for all depths, for one instantiation of the tissue.
    def __init__(self, contrasts, contrast_std_deviations, intensity_profile):
        self.depths = depths                                        # This is a list
        self.contrasts = contrasts                                  # This is a list
        self.contrast_std_deviations = contrast_std_deviations      # This is a list
        self.intensity_profile = intensity_profile                  # This is a list. TODO: Change variable name to intensity_profiles
        self.dx = dx_for_export
        self.wavelength = wavelength
        self.FDFD_dx = FDFD_dx
        self.FDFD_dz = FDFD_dz
        self.beam_radius = beam_radius
        self.focus_depth = focus_depth
        self.unique_layers = unique_layers
        self.n_h = n_h
        self.ls = ls
        self.g = g


def Tightfocus_HG(args):
    
    FDFD_depth = args[0]
    shared_mem_name = args[1]
    run_number = args[2]
    print('Simulation (HG) with depth %2.0f um, run number %2.0f starting...' %(10**6*FDFD_depth, run_number))
    
    beam_type = 'HG' # 'HG, 'LG', 'G'

    spot_size_at_start_of_FDFD_volume = SpotSizeCalculator(focus_depth,beam_radius,n_h,wavelength,FDFD_depth)
    xy_cells = optimal_cell_size(spot_size_at_start_of_FDFD_volume, FDFD_dx, min_xy_cells)

    # Parameters for saving the images to Results folder.
    dx = 5*beam_radius/(xy_cells)                 # Resolution for initial seed beam generation only.

    existing_shm = shared_memory.SharedMemory(name=shared_mem_name)
    n_global = np.ndarray((global_xy_cells,global_xy_cells,unique_layers), dtype=np.float32, buffer=existing_shm.buf)
    n = n_global[:xy_cells,:xy_cells,:]     # Generate a new view. This allocates no new memory

    (u,v) = (1,0)   # Mode numbers for HG beam
    seed_y = HG_beam(xy_cells, dx, beam_radius, u,v)    # This is well behaved and does not fill in at the focus
    seed_x = np.zeros(seed_y.shape)

    NA = n_h*1.5*beam_radius/focus_depth
    min_N = 4*NA**2*np.abs(FDFD_depth)/(np.sqrt(n_h**2-NA**2)*wavelength)

    if xy_cells<4*min_N:
        raise ValueError('Increase resolution!')

    Ex,Ey,Ez,_ = TightFocus(seed_x,seed_y,dx,wavelength,n_h,focus_depth,FDFD_depth,FDFD_dx,2040)
    Ex2,Ey2,Ez2,_ = TightFocus(seed_x,seed_y,dx,wavelength,n_h,focus_depth,FDFD_depth-FDFD_dz,FDFD_dx,2040)

    Uz = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
    Uy = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
    Ux = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)

    [Ux[:,:,0], Uy[:,:,0], Uz[:,:,0]] = [Ex, Ey, Ez]
    [Ux[:,:,1], Uy[:,:,1], Uz[:,:,1]] = [Ex2, Ey2, Ez2]

    Ux,Uy,Uz = Vector_FiniteDifference(Ux,Uy,Uz,FDFD_depth, FDFD_dx, FDFD_dz, xy_cells, n, wavelength, suppress_evanescent)
    HG10_Focus_Intensity = np.abs(Ux[:,:,2])**2+np.abs(Uy[:,:,2])**2+np.abs(Uz[:,:,2])**2

    #### Second HG beam ####

    (u,v) = (0,1)   # Mode numbers for HG beam
    seed_x = HG_beam(xy_cells, dx, beam_radius, u,v)    # This is well behaved and does not fill in at the focus
    seed_y = np.zeros(seed_x.shape)
    Ex,Ey,Ez,_ = TightFocus(seed_x,seed_y,dx,wavelength,n_h,focus_depth,FDFD_depth,FDFD_dx,2040)
    Ex2,Ey2,Ez2,_ = TightFocus(seed_x,seed_y,dx,wavelength,n_h,focus_depth,FDFD_depth-FDFD_dz,FDFD_dx,2040)

    [Ux[:,:,0], Uy[:,:,0], Uz[:,:,0]] = [Ex, Ey, Ez]
    [Ux[:,:,1], Uy[:,:,1], Uz[:,:,1]] = [Ex2, Ey2, Ez2]

    Ux,Uy,Uz = Vector_FiniteDifference(Ux,Uy,Uz,FDFD_depth, FDFD_dx, FDFD_dz, xy_cells, n, wavelength, suppress_evanescent)
    HG01_Focus_Intensity = np.abs(Ux[:,:,2])**2+np.abs(Uy[:,:,2])**2+np.abs(Uz[:,:,2])**2

    Focus_Intensity = HG01_Focus_Intensity+HG10_Focus_Intensity

    plot_HG(focus_depth,beam_radius,n_h,wavelength,xy_cells,FDFD_dx,HG10_Focus_Intensity,HG01_Focus_Intensity,Focus_Intensity,FDFD_depth,run_number)

    Contrast,Contrast_std_deviation = VortexNull(Focus_Intensity, FDFD_dx, beam_type, cross_sections = 19, num_samples = 1000)
    existing_shm.close()

    # For saving results
    original_axis = 10**6*FDFD_dx*np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
    export_axes = 10**6*dx_for_export*np.linspace(-exportResolution/2,exportResolution/2-1,exportResolution,dtype=np.int_)
    xx_export, yy_export = np.meshgrid(export_axes,export_axes, indexing='ij')  # Mesh grid used for exporting data
    export_field = RegularGridInterpolator((original_axis,original_axis),Focus_Intensity, bounds_error = True, method='linear')((xx_export, yy_export))

    print('Simulation (HG) with depth %2.0f um, run number %2.0f exiting' %(10**6*FDFD_depth, run_number))
    return Contrast, Contrast_std_deviation, export_field

def Tightfocus_LG(args):
    
    FDFD_depth = args[0]
    shared_mem_name = args[1]
    run_number = args[2]
    print('Simulation (LG) with depth %2.0f um, run number %2.0f starting...' %(10**6*FDFD_depth, run_number))

    beam_type = 'LG' # 'HG, 'LG', 'G'
    spot_size_at_start_of_FDFD_volume = SpotSizeCalculator(focus_depth,beam_radius,n_h,wavelength,FDFD_depth)
    xy_cells = optimal_cell_size(spot_size_at_start_of_FDFD_volume, FDFD_dx, min_xy_cells)

    # Parameters for saving the images to Results folder.
    dx = 5*beam_radius/(xy_cells)                 # Resolution for initial seed beam generation only.

    existing_shm = shared_memory.SharedMemory(name=shared_mem_name)
    n_global = np.ndarray((global_xy_cells,global_xy_cells,unique_layers), dtype=np.float32, buffer=existing_shm.buf)
    n = n_global[:xy_cells,:xy_cells,:]     # Generate a new view. This allocates no new memory

    l = 1
    seed_x = LG_OAM_beam(xy_cells, dx, beam_radius, l)
    seed_y = 1j*seed_x

    NA = n_h*1.5*beam_radius/focus_depth
    min_N = 4*NA**2*np.abs(FDFD_depth)/(np.sqrt(n_h**2-NA**2)*wavelength)

    if xy_cells<4*min_N:
        raise ValueError('Increase resolution!')

    Ex,Ey,Ez,_ = TightFocus(seed_x,seed_y,dx,wavelength,n_h,focus_depth,FDFD_depth,FDFD_dx,2040)
    Ex2,Ey2,Ez2,_ = TightFocus(seed_x,seed_y,dx,wavelength,n_h,focus_depth,FDFD_depth-FDFD_dz,FDFD_dx,2040)

    Uz = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
    Uy = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
    Ux = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)

    [Ux[:,:,0], Uy[:,:,0], Uz[:,:,0]] = [Ex, Ey, Ez]
    [Ux[:,:,1], Uy[:,:,1], Uz[:,:,1]] = [Ex2, Ey2, Ez2]

    Ux,Uy,Uz = Vector_FiniteDifference(Ux,Uy,Uz,FDFD_depth, FDFD_dx, FDFD_dz, xy_cells, n, wavelength, suppress_evanescent)
    LG_Focus_Intensity = np.abs(Ux[:,:,2])**2+np.abs(Uy[:,:,2])**2+np.abs(Uz[:,:,2])**2

    plot_LG(focus_depth,beam_radius,n_h,wavelength,xy_cells,FDFD_dx,LG_Focus_Intensity,FDFD_depth,run_number)

    Contrast,Contrast_std_deviation = VortexNull(LG_Focus_Intensity, FDFD_dx, beam_type, cross_sections = 19, num_samples = 1000)
    existing_shm.close()

    # For saving results
    original_axis = 10**6*FDFD_dx*np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
    export_axes = 10**6*dx_for_export*np.linspace(-exportResolution/2,exportResolution/2-1,exportResolution,dtype=np.int_)
    xx_export, yy_export = np.meshgrid(export_axes,export_axes, indexing='ij')  # Mesh grid used for exporting data
    export_field = RegularGridInterpolator((original_axis,original_axis),LG_Focus_Intensity, bounds_error = True, method='linear')((xx_export, yy_export))

    print('Simulation (LG) with depth %2.0f um, run number %2.0f exiting' %(10**6*FDFD_depth, run_number))
    return Contrast, Contrast_std_deviation, export_field


#### Test block ####
'''
if __name__ == '__main__':
    print('Cell size is ' + str(global_xy_cells)+'x'+str(global_xy_cells))
    print('NA of objective lens is '+str(n_h*beam_radius*1.5/focus_depth))
    shared_memory_bytes = int(global_xy_cells*global_xy_cells*unique_layers*4)
    shared_mem_name = 'Shared_test_block'
    try:
        shm = shared_memory.SharedMemory(name=shared_mem_name,create=True, size=shared_memory_bytes)
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=shared_mem_name,create=False, size=shared_memory_bytes)
    n_shared = np.ndarray((global_xy_cells,global_xy_cells,unique_layers), dtype='float32', buffer=shm.buf)
    n_shared[:,:,:]=RandomTissue([global_xy_cells, wavelength, FDFD_dx, FDFD_dz, n_h, ls, g, unique_layers, 0])
    # print(Tightfocus_LG([20e-6, shared_mem_name, 10]))
    Contrast, Contrast_std_deviation, _ = Tightfocus_LG([5e-6, shared_mem_name, 100])
    print(Contrast, Contrast_std_deviation)
    shm.unlink()
'''
if __name__ == '__main__':
    start_time = time.time()
    print('Cell size is ' + str(global_xy_cells)+'x'+str(global_xy_cells))
    print('NA of objective lens is '+str(n_h*beam_radius*1.5/focus_depth))
    shared_memory_bytes = int(global_xy_cells*global_xy_cells*unique_layers*4)  # float32 dtype: 4 bytes
    p = Pool(32)                # Remember! This executes everything outside this if statement!
    num_tissue_instances = 8    # Number of instances of tissue to generate and keep in memory. 2048x2048x70 grid takes 1.1 GB RAM. Minimal benefit to increasing beyond number of threads.
    num_runs = 40               # Number of runs. Keep this a multiple of num_tissue_instances.

    LG_result = []              # List of objects of class 'Results'
    HG_result = []
    tmp_contrasts_LG = np.zeros(len(depths))
    tmp_contrast_std_deviation_LG = np.zeros(len(depths))
    tmp_field_exports_LG  = [np.zeros((exportResolution,exportResolution))]*len(depths)
    tmp_contrasts_HG = np.zeros(len(depths))
    tmp_contrast_std_deviation_HG = np.zeros(len(depths))
    tmp_field_exports_HG  = [np.zeros((exportResolution,exportResolution))]*len(depths)
    
    run_number = 0
    random_seed = 0     # Run n should be seeded with seed = n, for reproducability.
    for iterator in range(int(num_runs/num_tissue_instances)):
        args = []
        shared_memory_blocks = []

        random_tissue_args = []
        for i in range(num_tissue_instances):
            random_seed = random_seed + 1
            random_tissue_args.append([global_xy_cells, wavelength, FDFD_dx, FDFD_dz, n_h, ls, g, unique_layers, random_seed])

        TissueModels = p.map(RandomTissue,random_tissue_args)
        for tissue_instance_number in range(num_tissue_instances):
            run_number = run_number+1
            shared_mem_name = 'TissueMatrix_'+str(tissue_instance_number)
            
            try:
                shared_memory_blocks.append(shared_memory.SharedMemory(name=shared_mem_name,create=True, size=shared_memory_bytes))
            except FileExistsError:
                shared_memory_blocks.append(shared_memory.SharedMemory(name=shared_mem_name,create=False, size=shared_memory_bytes))
            n_shared = np.ndarray((global_xy_cells,global_xy_cells,unique_layers), dtype='float32', buffer=shared_memory_blocks[-1].buf)
            n_shared[:,:,:]=TissueModels[tissue_instance_number]

            for depth in depths:
                args.append([depth, shared_mem_name, run_number])

        unrolled_results_LG = p.map(Tightfocus_LG, args)               # We need to roll up the results into lists, with each list containing the contrast for all depths for a given instance of tissue.
        unrolled_results_HG = p.map(Tightfocus_HG, args)
        print(str(int(100*(1+iterator)/int(num_runs/num_tissue_instances)))+' percent complete!')
        tmp_index = 0
        for tissue_instance_number in range(num_tissue_instances):  # For the sake of readbility, I'm not going to vectorize this step.
            for i in range(len(depths)):
                # This loop collects the resuls for all depths for a given tissue model into one list.
                tmp_contrasts_LG[i] = unrolled_results_LG[tmp_index][0]
                tmp_contrast_std_deviation_LG[i] = unrolled_results_LG[tmp_index][1]
                tmp_field_exports_LG[i] = unrolled_results_LG[tmp_index][2]

                tmp_contrasts_HG[i] = unrolled_results_HG[tmp_index][0]
                tmp_contrast_std_deviation_HG[i] = unrolled_results_HG[tmp_index][1]
                tmp_field_exports_HG[i] = unrolled_results_HG[tmp_index][2]

                tmp_index = tmp_index + 1

            # Garbage collector should automatically do this. However, some machines raise a memory leak warning if shared memory is not manually unlinked.
            shared_memory_blocks[tissue_instance_number].unlink()

            # Save results. The Results object is mutable in Python, so I need to deepcopy it.
            LG_result.append(copy.deepcopy(Results(tmp_contrasts_LG,tmp_contrast_std_deviation_LG,tmp_field_exports_LG)))
            HG_result.append(copy.deepcopy(Results(tmp_contrasts_HG,tmp_contrast_std_deviation_HG,tmp_field_exports_HG)))

    np.save('Results/Contrast_LG', LG_result)
    np.save('Results/Contrast_HG', HG_result)

    td = timedelta(seconds=time.time() - start_time)
    print('Time taken (hh:mm:ss):', td)