# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 11:37:25 2022

@author: thariq
"""
#%reset -sf
import numpy as np
import time
import matplotlib.pyplot as plt
from FriendlyFourierTransform import FFT2, iFFT2
from PropagationAlgorithm import propagate, propagate_Fourier, propagate_FiniteDifference
from SeedBeams import LG_OAM_beam, HG_beam, Gaussian_beam
from FieldPlots import PlotSnapshots, VortexNull
from GenerateRandomTissue import RandomTissue

start_time = time.time()

# Simulation parameters
wavelength = 500e-9
dz = 10e-9     
dx = dy = wavelength/6 # Minimum resolution = lambda/(n*sqrt(2)) for finite difference. Any lower and the algorithm is numerically unstable
ls = 15e-6  # Mean free path in tissue
g = 0.92    # Anisotropy factor
n_h = 1.33  # Homogenous part of refractive index
xy_cells = 1024     # Keep this a power of 2 for efficient FFT
absorption_padding = 2*dx # Thickness of absorbing boundary
Absorption_strength = 0.1   

beam_radius = 15e-6
focal_length = 30e-6/1.33 # Focal length of lens used to focus
Total_length = 30e-6


beam_type = 'G' # 'HG, 'LG', 'G'
use_picked_index = True
save_index = False
l = 1  # Topological charge for LG beam

if dz > (np.pi/10)**2*ls :
     NameError('Step size too large')


# Unique layers of biological tissue. Repeats the same random slice of tissue after this many steps.
# This leads to massive memory savings
unique_layers=int(Total_length/(5*dz))   

(u,v) = (1,0)   # Mode numbers for HG beam

plt.close('all')
imaging_depth = np.array([Total_length/3, 2*Total_length/3, Total_length]) # Take snapshots at these depths
print('Simulation volume is %1.1f um x %1.1f um x %1.1f um'  %(xy_cells*dx*10**6,xy_cells*dx*10**6,Total_length*10**6))

if xy_cells*dx < beam_radius*4:
    NameError('Simulation volume smaller than beam size')



indices = np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
xx, yy = np.meshgrid(dx*indices,dx*indices)


# We are solving second order diff. equation. Therefore we need two initial values.
# First intial value is the seed beam. 
# The second value is obtained by using Fast Fourier propagation algorithm to propagate seed by dz in water


k = 2*np.pi*n_h/wavelength
k0 = 2*np.pi/wavelength
dk = 2*np.pi/(dx*xy_cells)
kxkx, kyky = np.meshgrid(dk*indices,dk*indices)
H = np.exp(1j*dz*np.emath.sqrt((k)**2-kxkx**2-kyky**2))     # Fast Fourier propagator

Total_steps = int(Total_length/dz)+2
imaging_depth_indices = np.ndarray.astype((imaging_depth/dz),dtype=np.int_)
Field_snapshots = np.zeros((xy_cells,xy_cells,1+np.size(imaging_depth_indices)),dtype=np.complex64)


U = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
A = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)


## Setting up the medium
print('Generating randomized tissue index')

if use_picked_index:
    n = np.load('refractive_index.npy')
    print('Warning: Loading picked refractive index')
    
else:
    n = RandomTissue(xy_cells, Total_steps, wavelength, dx, dz, n_h, ls, g,unique_layers)
print('Done!')



## Setting up beam
if beam_type=='LG':
    U[:,:,0] = LG_OAM_beam(xy_cells, dx, beam_radius, l)
elif beam_type=='HG':
    U[:,:,0] = HG_beam(xy_cells, dx, beam_radius, u,v)
else:
    U[:,:,0] = Gaussian_beam(xy_cells, dx, beam_radius)
    

# Focus the beam by applying a parabolic phase like a lens would do
U[:,:,0] = U[:,:,0]*np.exp(1j*k0*(-(xx**2+yy**2)/(2*focal_length)))
seed = U[:,:,0].copy()



## For first two steps of U
A[:,:,0] = FFT2(U[:,:,0])
A[:,:,1] = H*A[:,:,0]
U[:,:,1] = iFFT2(A[:,:,1])
current_step = 2

# propagate_Fourier() does standard Fourier Beam Propagation
# propagate() does finite difference propagation. The step size needs to be much smaller.

U,A, Field_snapshots, current_step = propagate(U, A,Total_length, current_step, dx, dz, xy_cells, n, imaging_depth_indices, absorption_padding, Absorption_strength, wavelength)
Field_snapshots[:,:,0] = seed

if beam_type=='LG' or beam_type=='G':
    for i in range(np.shape(Field_snapshots)[2]):
        # Convert electric field to intensity
        Field_snapshots[:,:,i] = np.abs(Field_snapshots[:,:,i])**2
    f = PlotSnapshots(Field_snapshots, imaging_depth)
    f.suptitle(beam_type+ str(l))
    f.show()
elif beam_type=='HG':
    # We are interested in forming a donut by combining incoherent HG10 and HG01 beams
    # The profile of HG01 is obtained by transposing HG10
    for i in range(np.shape(Field_snapshots)[2]):
        Field_snapshots[:,:,i] = (np.abs(Field_snapshots[:,:,i])**2+np.abs(Field_snapshots[:,:,i].transpose())**2)
    f = PlotSnapshots(Field_snapshots, imaging_depth)    
    f.suptitle(beam_type+ str(u)+str(v))


print('Sanity test: Total power at various depth')
print('Power of seed: %1.4f' %(float(np.sum(Field_snapshots[:,:,0]*dx**2))))
print('Power at depth 1: %1.4f' %(np.sum(Field_snapshots[:,:,1]*dx**2)))
print('Power at depth 2: %1.4f' %(np.sum(Field_snapshots[:,:,2]*dx**2)))
print('Power at depth 3: %1.4f' %(np.sum(Field_snapshots[:,:,3]*dx**2)))

VNull = VortexNull(Field_snapshots[:,:,3], dx, beam_type, cross_sections = 19, num_samples = 1000)
print("--- %s seconds ---" % '%.2f'%(time.time() - start_time))

if save_index:
    with open('refractive_index.npy', 'wb') as f:
        np.save(f, n)