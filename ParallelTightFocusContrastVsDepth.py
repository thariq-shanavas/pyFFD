import numpy as np
import time
import matplotlib.pyplot as plt
from PropagationAlgorithm import Vector_FiniteDifference
from SeedBeams import LG_OAM_beam, HG_beam
from FieldPlots import VortexNull
from GenerateRandomTissue import RandomTissue
from DebyeWolfIntegral import TightFocus, SpotSizeCalculator
from multiprocessing import Pool, shared_memory
from scipy.interpolate import RegularGridInterpolator



# Simulation parameters
beam_radius = 1e-3
focus_depth = 3.5e-3    # Depth at which the beam is focused. Note that this is not the focal length in air.
depths = np.array([35e-6,25e-6,15e-6,5e-6])      # Calculate the contrast at these tissue depths
# depths = 1e-6*np.array([10, 5])      # Calculate the contrast at these tissue depths
n_h = 1.33  # Homogenous part of refractive index
ls = 15e-6  # Mean free path in tissue
g = 0.92    # Anisotropy factor
dz = 25e-9


xy_cells = 256    # Keep this a power of 2 for efficient FFT
# shared_mem_name = 'TissueMatrix2048x30'     # Just a name for a shared memory space.
wavelength = 500e-9


# Expected spot size (1/e^2 diameter) at beginning of numerical simulation volume
# We need to keep the dx same for all tissue depths to avoid bias from sampling resolution.
# So we use the smallest dx that would work for all depths we are interested in.

max_spot_size_at_start_of_FDFD_volume = SpotSizeCalculator(focus_depth,beam_radius,n_h,wavelength,np.max(depths))
FDFD_dx = max_spot_size_at_start_of_FDFD_volume*2/xy_cells   # Target dx for debye-wolf calc output

spot_size_at_focus = SpotSizeCalculator(focus_depth,beam_radius,n_h,wavelength,0)   # For plotting
imaging_dx = spot_size_at_focus*6/xy_cells
indices = np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
axis = 10**6*FDFD_dx*indices
imaging_axes = 10**6*imaging_dx*indices
xx_imaging, yy_imaging = np.meshgrid(imaging_axes,imaging_axes, indexing='ij')  # Mesh grid used for plotting

if dz/FDFD_dx > 0.5 or dz>wavelength/10:
    raise ValueError('Reduce dz!')

unique_layers = 70    # Unique layers of index for procedural generation of tissue index. Unclear what's the effect of making this small.

# Other parameters - do not change
dx = 5*beam_radius/(xy_cells) 

if FDFD_dx > wavelength/((n_h+0.1)*1.41):     # If resolution > lambda/sqrt(2), Evanescent fields blow up. 0.1 adds a small margin of error.
    suppress_evanescent = False
else:
    suppress_evanescent = True

def Tightfocus_HG(args):
    
    fig = plt.figure()
    plt.gcf().set_dpi(500)
    plt.rcParams.update({'font.size': 5})
    plt.rcParams['pcolor.shading'] = 'auto'
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    #plt.rcParams["axes.linewidth"] = 2
    
    ax1 = fig.add_subplot(1,3,1, adjustable='box', aspect=1)
    ax2 = fig.add_subplot(1,3,2, adjustable='box', aspect=1)
    ax3 = fig.add_subplot(1,3,3, adjustable='box', aspect=1)
    
    FDFD_depth = args[0]
    shared_mem_name = args[1]
    run_number = args[2]
    # print('Thread running with depth %1.2f um, run number %1.2f' %(10**6*FDFD_depth, run_number))
    
    beam_type = 'HG' # 'HG, 'LG', 'G'

    existing_shm = shared_memory.SharedMemory(name=shared_mem_name)
    n = np.ndarray((xy_cells,xy_cells,unique_layers), dtype=np.float32, buffer=existing_shm.buf)

    (u,v) = (1,0)   # Mode numbers for HG beam
    seed_y = HG_beam(xy_cells, dx, beam_radius, u,v)    # This is well behaved and does not fill in at the focus
    seed_x = np.zeros(seed_y.shape)

    NA = n_h*1.5*beam_radius/focus_depth
    min_N = 4*NA**2*np.abs(FDFD_depth)/(np.sqrt(n_h**2-NA**2)*wavelength)

    if xy_cells<4*min_N:
        raise ValueError('Increase resolution!')

    Ex,Ey,Ez,_ = TightFocus(seed_x,seed_y,dx,wavelength,n_h,focus_depth,FDFD_depth,FDFD_dx,4096)
    Ex2,Ey2,Ez2,_ = TightFocus(seed_x,seed_y,dx,wavelength,n_h,focus_depth,FDFD_depth-dz,FDFD_dx,4096)

    Uz = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
    Uy = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
    Ux = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)

    Uz[:,:,0] = Ez
    Uz[:,:,1] = Ez2
    Uy[:,:,0] = Ey
    Uy[:,:,1] = Ey2
    Ux[:,:,0] = Ex
    Ux[:,:,1] = Ex2

    Ux,Uy,Uz = Vector_FiniteDifference(Ux,Uy,Uz,FDFD_depth, FDFD_dx, dz, xy_cells, n, wavelength, suppress_evanescent)
    HG10_Focus_Intensity = np.abs(Ux[:,:,2])**2+np.abs(Uy[:,:,2])**2+np.abs(Uz[:,:,2])**2

    ax1.pcolormesh(imaging_axes,imaging_axes,RegularGridInterpolator((axis,axis),HG10_Focus_Intensity, bounds_error = True, method='linear')((xx_imaging, yy_imaging)))
    ax1.set_title('HG 10 beam at focus', fontweight='bold')
    ax1.set_xlabel("x ($µm$)", fontweight='bold')
    ax1.set_ylabel("y ($µm$)", fontweight='bold')
        
    #### Second HG beam ####

    (u,v) = (0,1)   # Mode numbers for HG beam
    seed_x = HG_beam(xy_cells, dx, beam_radius, u,v)    # This is well behaved and does not fill in at the focus
    seed_y = np.zeros(seed_x.shape)
    Ex,Ey,Ez,_ = TightFocus(seed_x,seed_y,dx,wavelength,n_h,focus_depth,FDFD_depth,FDFD_dx,4096)
    Ex2,Ey2,Ez2,_ = TightFocus(seed_x,seed_y,dx,wavelength,n_h,focus_depth,FDFD_depth-dz,FDFD_dx,4096)

    Uz[:,:,0] = Ez
    Uz[:,:,1] = Ez2
    Uy[:,:,0] = Ey
    Uy[:,:,1] = Ey2
    Ux[:,:,0] = Ex
    Ux[:,:,1] = Ex2

    Ux,Uy,Uz = Vector_FiniteDifference(Ux,Uy,Uz,FDFD_depth, FDFD_dx, dz, xy_cells, n, wavelength, suppress_evanescent)
    HG01_Focus_Intensity = np.abs(Ux[:,:,2])**2+np.abs(Uy[:,:,2])**2+np.abs(Uz[:,:,2])**2

    ax2.pcolormesh(imaging_axes,imaging_axes,RegularGridInterpolator((axis,axis),HG01_Focus_Intensity, bounds_error = True, method='linear')((xx_imaging, yy_imaging)))
    ax2.set_title('HG 01 beam at focus', fontweight='bold')
    ax2.set_xlabel("x ($µm$)", fontweight='bold')
    ax2.set_ylabel("y ($µm$)", fontweight='bold')

    Focus_Intensity = HG01_Focus_Intensity+HG10_Focus_Intensity
    ax3.pcolormesh(imaging_axes,imaging_axes,RegularGridInterpolator((axis,axis),Focus_Intensity , bounds_error = True, method='linear')((xx_imaging, yy_imaging)))
    ax3.set_title('HG 01 + HG 10 (Incoherent Donut)', fontweight='bold')
    ax3.set_xlabel("x ($µm$)", fontweight='bold')
    ax3.set_ylabel("y ($µm$)", fontweight='bold')

    plt.tight_layout()
    plt.savefig('Results/HG_'+str("{:02d}".format(int(1e6*FDFD_depth)))+'um_run'+str("{:02d}".format(run_number))+'.png')
    plt.close()

    Contrast,_ = VortexNull(Focus_Intensity, FDFD_dx, beam_type, cross_sections = 19, num_samples = 1000)
    return Contrast

def Tightfocus_LG(args):
    
    fig = plt.figure()
    plt.gca().set_aspect('equal')
    plt.gcf().set_dpi(500)
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['pcolor.shading'] = 'auto'
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.linewidth"] = 2
    
    FDFD_depth = args[0]
    shared_mem_name = args[1]
    run_number = args[2]
    
    beam_type = 'LG' # 'HG, 'LG', 'G'

    existing_shm = shared_memory.SharedMemory(name=shared_mem_name)
    n = np.ndarray((xy_cells,xy_cells,unique_layers), dtype=np.float32, buffer=existing_shm.buf)

    l = 1
    seed_x = LG_OAM_beam(xy_cells, dx, beam_radius, l)
    seed_y = 1j*seed_x

    NA = n_h*1.5*beam_radius/focus_depth
    min_N = 4*NA**2*np.abs(FDFD_depth)/(np.sqrt(n_h**2-NA**2)*wavelength)

    if xy_cells<4*min_N:
        raise ValueError('Increase resolution!')

    Ex,Ey,Ez,_ = TightFocus(seed_x,seed_y,dx,wavelength,n_h,focus_depth,FDFD_depth,FDFD_dx,4096)
    Ex2,Ey2,Ez2,_ = TightFocus(seed_x,seed_y,dx,wavelength,n_h,focus_depth,FDFD_depth-dz,FDFD_dx,4096)

    Uz = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
    Uy = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
    Ux = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)

    Uz[:,:,0] = Ez
    Uz[:,:,1] = Ez2
    Uy[:,:,0] = Ey
    Uy[:,:,1] = Ey2
    Ux[:,:,0] = Ex
    Ux[:,:,1] = Ex2

    Ux,Uy,Uz = Vector_FiniteDifference(Ux,Uy,Uz,FDFD_depth, FDFD_dx, dz, xy_cells, n, wavelength, suppress_evanescent)
    LG_Focus_Intensity = np.abs(Ux[:,:,2])**2+np.abs(Uy[:,:,2])**2+np.abs(Uz[:,:,2])**2

    plt.pcolormesh(imaging_axes,imaging_axes,RegularGridInterpolator((axis,axis),LG_Focus_Intensity, bounds_error = True, method='linear')((xx_imaging, yy_imaging)))
    plt.title("LG beam at focus", weight='bold')
    plt.xlabel("x ($µm$)", weight='bold', fontsize=12)
    plt.xticks(weight = 'bold', fontsize=12)
    plt.ylabel("y ($µm$)", weight='bold', fontsize=12)
    plt.yticks(weight = 'bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('Results/LG_'+str("{:02d}".format(int(1e6*FDFD_depth)))+'um_run'+str("{:02d}".format(run_number))+'.png')
    plt.close()

    Contrast,_ = VortexNull(LG_Focus_Intensity, FDFD_dx, beam_type, cross_sections = 19, num_samples = 1000)
    return Contrast


#### Test block ####
shared_memory_bytes = int(xy_cells*xy_cells*unique_layers*4)
shared_mem_name = 'Shared_test_block'
try:
    shm = shared_memory.SharedMemory(name=shared_mem_name,create=True, size=shared_memory_bytes)
except FileExistsError:
    shm = shared_memory.SharedMemory(name=shared_mem_name,create=False, size=shared_memory_bytes)
n_shared = np.ndarray((xy_cells,xy_cells,unique_layers), dtype='float32', buffer=shm.buf)
n_shared[:,:,:]=RandomTissue(xy_cells, wavelength, FDFD_dx, dz, n_h, ls, g, unique_layers)
Tightfocus_LG([5e-6, shared_mem_name, 10])

'''
if __name__ == '__main__':
    start_time = time.time()
    shared_memory_bytes = int(xy_cells*xy_cells*unique_layers*4)  # float32 dtype: 4 bytes
    p = Pool(12)                # Remember! This executes everything outside this if statement!
    num_tissue_instances = 4    # Number of instances of tissue to generate and keep in memory. 2048x2048x70 grid takes 1.1 GB RAM. Minimal benefit to increasing beyond number of threads.
    num_runs = 20               # Number of runs. Keep this a multiple of num_tissue_instances.

    LG_result = np.zeros((6,num_runs))
    HG_result = np.zeros((num_tissue_instances*len(depths),num_runs))
    
    run_number = 0
    for iterator in range(int(num_runs/num_tissue_instances)):
        args = []
        # This is necessary to prevent Python garbage collector from prematurely deleting shared memory blocks.
        # The second time this line runs, it asks the garbage collector to release the shared memory used in the previous loop.
        shared_memory_blocks = []

        for tissue_instance_number in range(num_tissue_instances):
            run_number = run_number+1
            shared_mem_name = 'TissueMatrix_'+str(tissue_instance_number)
            
            try:
                shared_memory_blocks.append(shared_memory.SharedMemory(name=shared_mem_name,create=True, size=shared_memory_bytes))
            except FileExistsError:
                shared_memory_blocks.append(shared_memory.SharedMemory(name=shared_mem_name,create=False, size=shared_memory_bytes))
            n_shared = np.ndarray((xy_cells,xy_cells,unique_layers), dtype='float32', buffer=shared_memory_blocks[-1].buf)
            n_shared[:,:,:]=RandomTissue(xy_cells, wavelength, FDFD_dx, dz, n_h, ls, g, unique_layers)

            for depth in depths:
                args.append([depth, shared_mem_name, run_number])
        
        
        LG_result[:,run_number] = p.map(Tightfocus_LG, args)
        print('LG results')
        print(LG_result[:,run_number])

        #HG_result[:,run_number] = p.map(Tightfocus_HG, args)
        #print('HG results')
        #print(HG_result[:,run_number])
    
    print('LG results final')
    print(LG_result)

    print('HG results final')
    print(HG_result)

    np.save('Contrast_LG', LG_result)
    np.save('Contrast_HG', HG_result)


    print("--- %s seconds ---" % '%.2f'%(time.time() - start_time))
    '''