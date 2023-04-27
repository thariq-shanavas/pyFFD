import numpy as np
import time
import matplotlib.pyplot as plt
from FriendlyFourierTransform import FFT2, iFFT2
from PropagationAlgorithm import propagate, propagate_Fourier, propagate_FiniteDifference
from SeedBeams import LG_OAM_beam, HG_beam, Gaussian_beam
from FieldPlots import PlotSnapshots, VortexNull
from GenerateRandomTissue import RandomTissue
from DebyeWolfIntegral import TightFocus, SpotSizeCalculator

start_time = time.time()
propagation_algorithm = propagate_FiniteDifference
suppress_evanescent = True
beam_type = 'LG' # 'HG, 'LG', 'G'
depths = 1e-6*np.array([5,15,25,35,45,55])

Contrast = np.zeros(depths.shape)
radii = np.zeros(depths.shape)

# Simulation parameters
beam_radius = 1e-3
focus_depth = 3.5e-3

xy_cells = 1024    # Keep this a power of 2 for efficient FFT
wavelength = 500e-9
target_dx = 817e-10   # Target dx for debye-wolf calc output
dz = 10e-9
n_h = 1.33  # Homogenous part of refractive index
ls = 15e-6  # Mean free path in tissue
g = 0.92    # Anisotropy factor
unique_layers=100
n = np.load('refractive_index_1024_500_817e-10_10e-9_133e-2_15e-6_92e-2_100.npy')


dx = 5*beam_radius/(xy_cells) 
absorption_padding = 5*dx # Thickness of absorbing boundary
Absorption_strength = 10


l = 1  # Topological charge for LG beam
(u,v) = (1,0)   # Mode numbers for HG beam


if beam_type=='LG':
    seed_x = LG_OAM_beam(xy_cells, dx, beam_radius, l)
    seed_y = 1j*seed_x
    #seed_y = np.zeros(seed_x.shape)
elif beam_type=='HG':
    seed_y = HG_beam(xy_cells, dx, beam_radius, u,v)    # This is well behaved and does not fill in at the focus
    seed_x = np.zeros(seed_y.shape)
elif beam_type=='G':
    seed_x = Gaussian_beam(xy_cells, dx, beam_radius)
    seed_y = np.zeros(seed_x.shape)
else:
    xx,yy = np.meshgrid(dx*np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_),dx*np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_))
    seed_x = (xx**2+yy**2<beam_radius**2).astype(float)
    seed_y = np.zeros(seed_x.shape)

#n = RandomTissue(xy_cells, wavelength, target_dx, dz, n_h, ls, g, unique_layers)

for run_number in range(depths.size):
    FDFD_depth = depths[run_number]
    NA = n_h*1.5*beam_radius/focus_depth
    min_N = 4*NA**2*np.abs(FDFD_depth)/(np.sqrt(n_h**2-NA**2)*wavelength)
    print('Minimum samples: %1.0f' %(4*min_N))
    expected_spot_size = SpotSizeCalculator(focus_depth,beam_radius,n_h,wavelength,FDFD_depth)  # Expected spot size (1/e^2 diameter) at beginning of numerical simulation volume

    if xy_cells<4*min_N:
        raise ValueError('Increase resolution!')

    Ex,Ey,Ez,_ = TightFocus(seed_x,seed_y,dx,wavelength,n_h,focus_depth,FDFD_depth,target_dx,4096)
    Ex2,Ey2,Ez2,_ = TightFocus(seed_x,seed_y,dx,wavelength,n_h,focus_depth,FDFD_depth-dz,target_dx,4096)
    print('Discretization changed from %1.1f nm to %1.1f nm'  %(dx*10**9,target_dx*10**9))

    unique_layers=100
    Total_steps = int(FDFD_depth/dz)+2
    imaging_depth = [] # Take snapshots at these depths
    imaging_depth_indices = []
    

    Uz = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
    Az = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
    Uy = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
    Ay = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
    Ux = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
    Ax = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)

    Uz[:,:,0] = Ez
    Az[:,:,0] = FFT2(Uz[:,:,0])
    Uz[:,:,1] = Ez2
    Az[:,:,1] = FFT2(Uz[:,:,1])
    Uy[:,:,0] = Ey
    Ay[:,:,0] = FFT2(Uz[:,:,0])
    Uy[:,:,1] = Ey2
    Ay[:,:,1] = FFT2(Uz[:,:,1])
    Ux[:,:,0] = Ex
    Ax[:,:,0] = FFT2(Uz[:,:,0])
    Ux[:,:,1] = Ex2
    Ax[:,:,1] = FFT2(Uz[:,:,1])


    current_step = 2
    Uz,Az, _, _ = propagation_algorithm(Uz, Az,FDFD_depth, current_step, target_dx, dz, xy_cells, n, imaging_depth_indices, absorption_padding, Absorption_strength, wavelength, suppress_evanescent)
    Uy,Ay, _, _ = propagation_algorithm(Uy, Ay,FDFD_depth, current_step, target_dx, dz, xy_cells, n, imaging_depth_indices, absorption_padding, Absorption_strength, wavelength, suppress_evanescent)
    Ux,Ax, _, _ = propagation_algorithm(Ux, Ax,FDFD_depth, current_step, target_dx, dz, xy_cells, n, imaging_depth_indices, absorption_padding, Absorption_strength, wavelength, suppress_evanescent)

    if beam_type=='HG':
        Focus_Intensity = np.abs(Ux[:,:,2])**2+np.abs(Uy[:,:,2])**2+np.abs(Uz[:,:,2])**2
        Focus_Intensity = (Focus_Intensity+np.transpose(Focus_Intensity))/2
    else:
        Focus_Intensity = np.abs(Ux[:,:,2])**2+np.abs(Uy[:,:,2])**2+np.abs(Uz[:,:,2])**2

    Contrast[run_number],radii[run_number] = VortexNull(Focus_Intensity, target_dx, beam_type, cross_sections = 19, num_samples = 1000)

import seaborn as sns
sns.heatmap(np.abs(seed_x))
sns.heatmap(np.abs(seed_y))
sns.heatmap(np.abs(Ex))
sns.heatmap(np.abs(Ey))
sns.heatmap(np.abs(Ez))
sns.heatmap(np.abs(Ux[:,:,1]))
sns.heatmap(np.abs(Uy[:,:,1]))
sns.heatmap(np.abs(Uz[:,:,1]))
np.save('Contrast_LG', Contrast)
np.save('Radii_LG', radii)
print("--- %s seconds ---" % '%.2f'%(time.time() - start_time))