import numpy as np
import time
import matplotlib.pyplot as plt
from FriendlyFourierTransform import FFT2, iFFT2
from PropagationAlgorithm import propagate, propagate_Fourier, propagate_FiniteDifference
from SeedBeams import LG_OAM_beam, HG_beam, Gaussian_beam
from FieldPlots import PlotSnapshots, VortexNull
from GenerateRandomTissue import RandomTissue
from DebyeWolfIntegral import TightFocus, SpotSizeCalculator
from multiprocessing import Pool, shared_memory

plt.rcParams['figure.dpi']= 1800
plt.rcParams.update({'font.size': 4})
plt.rcParams['pcolor.shading'] = 'auto'

xy_cells = 2048    # Keep this a power of 2 for efficient FFT
wavelength = 500e-9
target_dx = 408e-10   # Target dx for debye-wolf calc output


def parallel_tightfocus(depth):
    FDFD_depth = depth
    propagation_algorithm = propagate_FiniteDifference
    suppress_evanescent = True
    beam_type = 'LG' # 'HG, 'LG', 'G'

    # Simulation parameters
    beam_radius = 1e-3
    focus_depth = 3.5e-3

    dz = 10e-9
    n_h = 1.33  # Homogenous part of refractive index
    ls = 15e-6  # Mean free path in tissue
    g = 0.92    # Anisotropy factor
    unique_layers=100
    #n = RandomTissue(xy_cells, wavelength, target_dx, dz, n_h, ls, g, unique_layers)
    #n = np.load('refractive_index_1024_500_817e-10_10e-9_133e-2_15e-6_92e-2_100.npy')
    existing_shm = shared_memory.SharedMemory(name='TissueMatrix2048x100')
    n = np.ndarray((2048,2048,100), dtype=np.float32, buffer=existing_shm.buf)

    dx = 5*beam_radius/(xy_cells) 
    absorption_padding = 5*dx # Thickness of absorbing boundary
    Absorption_strength = 10

    (u,v) = (1,0)   # Mode numbers for HG beam
    seed_y = HG_beam(xy_cells, dx, beam_radius, u,v)    # This is well behaved and does not fill in at the focus
    seed_x = np.zeros(seed_y.shape)
            
    NA = n_h*1.5*beam_radius/focus_depth
    min_N = 4*NA**2*np.abs(FDFD_depth)/(np.sqrt(n_h**2-NA**2)*wavelength)
    print('Minimum samples: %1.0f' %(4*min_N))
    expected_spot_size = SpotSizeCalculator(focus_depth,beam_radius,n_h,wavelength,FDFD_depth)  # Expected spot size (1/e^2 diameter) at beginning of numerical simulation volume
    print('Tissue depth %1.1f um. Expected spot size at start of FDFD region is %1.2f um, simulation cross section is %1.2f um' %(10**6*depth, 10**6*expected_spot_size,xy_cells*target_dx*10**6))

    if xy_cells<4*min_N:
        raise ValueError('Increase resolution!')

    Ex,Ey,Ez,_ = TightFocus(seed_x,seed_y,dx,wavelength,n_h,focus_depth,FDFD_depth,target_dx,4096)
    Ex2,Ey2,Ez2,_ = TightFocus(seed_x,seed_y,dx,wavelength,n_h,focus_depth,FDFD_depth-dz,target_dx,4096)
    print('Discretization changed from %1.1f nm to %1.1f nm'  %(dx*10**9,target_dx*10**9))

    imaging_depth = [] # Take snapshots at these depths
    imaging_depth_indices = []

    Uz = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
    Uy = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
    Ux = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)

    Uz[:,:,0] = Ez
    Uz[:,:,1] = Ez2
    Uy[:,:,0] = Ey
    Uy[:,:,1] = Ey2
    Ux[:,:,0] = Ex
    Ux[:,:,1] = Ex2

    current_step = 2
    Uz,_, _, _ = propagation_algorithm(Uz, 0,FDFD_depth, current_step, target_dx, dz, xy_cells, n, imaging_depth_indices, absorption_padding, Absorption_strength, wavelength, suppress_evanescent)
    Uy,_, _, _ = propagation_algorithm(Uy, 0,FDFD_depth, current_step, target_dx, dz, xy_cells, n, imaging_depth_indices, absorption_padding, Absorption_strength, wavelength, suppress_evanescent)
    Ux,_, _, _ = propagation_algorithm(Ux, 0,FDFD_depth, current_step, target_dx, dz, xy_cells, n, imaging_depth_indices, absorption_padding, Absorption_strength, wavelength, suppress_evanescent)

    Focus_Intensity = np.abs(Ux[:,:,2])**2+np.abs(Uy[:,:,2])**2+np.abs(Uz[:,:,2])**2

    
    indices = np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
    axis = 10**6*target_dx*indices
    fig = plt.figure()
    ax1 = fig.add_subplot(1,3,1, adjustable='box', aspect=1)
    ax2 = fig.add_subplot(1,3,2, adjustable='box', aspect=1)
    ax3 = fig.add_subplot(1,3,3, adjustable='box', aspect=1)
    
    ax1.pcolormesh(axis,axis,np.abs(Focus_Intensity))
    ax1.title.set_text("HG 10 at focus")
        
    #### Second HG beam ####

    (u,v) = (0,1)   # Mode numbers for HG beam
    seed_x = HG_beam(xy_cells, dx, beam_radius, u,v)    # This is well behaved and does not fill in at the focus
    seed_y = np.zeros(seed_y.shape)
    Ex,Ey,Ez,_ = TightFocus(seed_x,seed_y,dx,wavelength,n_h,focus_depth,FDFD_depth,target_dx,4096)
    Ex2,Ey2,Ez2,_ = TightFocus(seed_x,seed_y,dx,wavelength,n_h,focus_depth,FDFD_depth-dz,target_dx,4096)

    Uz[:,:,0] = Ez
    Uz[:,:,1] = Ez2
    Uy[:,:,0] = Ey
    Uy[:,:,1] = Ey2
    Ux[:,:,0] = Ex
    Ux[:,:,1] = Ex2

    Uz,_, _, _ = propagation_algorithm(Uz, 0,FDFD_depth, current_step, target_dx, dz, xy_cells, n, imaging_depth_indices, absorption_padding, Absorption_strength, wavelength, suppress_evanescent)
    Uy,_, _, _ = propagation_algorithm(Uy, 0,FDFD_depth, current_step, target_dx, dz, xy_cells, n, imaging_depth_indices, absorption_padding, Absorption_strength, wavelength, suppress_evanescent)
    Ux,_, _, _ = propagation_algorithm(Ux, 0,FDFD_depth, current_step, target_dx, dz, xy_cells, n, imaging_depth_indices, absorption_padding, Absorption_strength, wavelength, suppress_evanescent)

    ax2.pcolormesh(axis,axis,np.abs(Ux[:,:,2])**2+np.abs(Uy[:,:,2])**2+np.abs(Uz[:,:,2])**2)
    ax2.title.set_text("HG01 at focus")

    Focus_Intensity = Focus_Intensity + np.abs(Ux[:,:,2])**2+np.abs(Uy[:,:,2])**2+np.abs(Uz[:,:,2])**2

    ax3.pcolormesh(axis,axis,Focus_Intensity)
    ax3.title.set_text("Incoherent Donut")

    plt.savefig('HG_'+str(int(1e6*depth))+'.png')
    plt.close()

    Contrast,_ = VortexNull(Focus_Intensity, target_dx, beam_type, cross_sections = 19, num_samples = 1000)
    return Contrast


if __name__ == '__main__':
    start_time = time.time()
    depths = 1e-6*np.array([55,45,35,25,15,5])
    n = np.load('refractive_index_2048_500_408e-10_10e-9_133e-2_15e-6_92e-2_100.npy')
    try:
        shm = shared_memory.SharedMemory(name='TissueMatrix2048x100',create=True, size=n.nbytes)
    except FileExistsError:
        shm = shared_memory.SharedMemory(name='TissueMatrix2048x100',create=False, size=n.nbytes)
    n_shared = np.ndarray(n.shape, dtype=n.dtype, buffer=shm.buf)
    n_shared[:,:,:]=n
    del n

    p = Pool(2)
    result = p.map(parallel_tightfocus, depths)
    print(result)
    print("--- %s seconds ---" % '%.2f'%(time.time() - start_time))
    np.save('Contrast_LG', result)
