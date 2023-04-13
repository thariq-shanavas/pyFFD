import numpy as np
import time
import matplotlib.pyplot as plt
from FriendlyFourierTransform import FFT2, iFFT2
from PropagationAlgorithm import propagate, propagate_Fourier
from SeedBeams import LG_OAM_beam, HG_beam, Gaussian_beam
from FieldPlots import PlotSnapshots, VortexNull
from GenerateRandomTissue import RandomTissue
from DebyeWolfIntegral import TightFocus


start_time = time.time()
plt.rcParams['figure.dpi']= 300
plt.rcParams.update({'font.size': 4})
plt.rcParams['pcolor.shading'] = 'auto'

# Simulation parameters
beam_radius = 100e-6
focus_depth = 1e-3
FDFD_depth = 50e-6 #5e-6       # Debye-Wolf integral to calculate field at focus_depth-FDFD_depth, then FDFD to focus

wavelength = 500e-9
xy_cells = 512    # Keep this a power of 2 for efficient FFT
dz = 80e-9
dx = dy = 10*(2*beam_radius)/(xy_cells) # Minimum resolution = lambda/(n*sqrt(2)) for finite difference. Any lower and the algorithm is numerically unstable
# Note that the dx changes after the tight focus. Make sure the dx is still greater than lambda/(n*sqrt(2))

absorption_padding = 3*dx # Thickness of absorbing boundary
Absorption_strength = 0.25
n_h = 1.33  # Homogenous part of refractive index

expected_spot_size = 15e-6  # Expected spot size (1/e^2 diameter) at beginning of numerical simulation volume
target_dx = 3*expected_spot_size/xy_cells   # Target dx for debye-wolf calc output


ls = 15e-6  # Mean free path in tissue
g = 0.92    # Anisotropy factor

if 2*beam_radius > 0.5*dx*xy_cells:
    # Beam diameter greater than half the length of the simulation cross section.
    ValueError("Beam is larger than simulation cross section")
if dz > (np.pi/10)**2*ls :
    NameError('Step size too large')



beam_type = 'G' # 'HG, 'LG', 'G'
l = 1  # Topological charge for LG beam
(u,v) = (1,0)   # Mode numbers for HG beam

if beam_type=='LG':
    seed = LG_OAM_beam(xy_cells, dx, beam_radius, l)
elif beam_type=='HG':
    seed = HG_beam(xy_cells, dx, beam_radius, u,v)
else:
    seed = Gaussian_beam(xy_cells, dx, beam_radius)

unique_layers=50
print('Simulation volume is %1.1f um x %1.1f um x %1.1f um'  %(xy_cells*dx*10**6,xy_cells*dx*10**6,focus_depth*10**6))

# Calculate fields at FDFD_depth
dx_seed = dx
Ex,Ey,Ez,_ = TightFocus(seed,dx_seed,wavelength,n_h,focus_depth,FDFD_depth,target_dx)
Ex2,Ey2,Ez2,dx = TightFocus(seed,dx_seed,wavelength,n_h,focus_depth,FDFD_depth-dz,target_dx)
print('Discretization changed from %1.1f nm to %1.1f nm'  %(dx_seed*10**9,dx*10**9))


# FDFD Propagation
Total_steps = int(FDFD_depth/dz)+2
imaging_depth = [] # Take snapshots at these depths
imaging_depth_indices = []

indices = np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
#xx, yy = np.meshgrid(dx*indices,dx*indices)
#k = 2*np.pi*n_h/wavelength
#k0 = 2*np.pi/wavelength
#dk = 2*np.pi/(dx*xy_cells)
#kxkx, kyky = np.meshgrid(dk*indices,dk*indices)
#H = np.exp(1j*dz*np.emath.sqrt((k)**2-kxkx**2-kyky**2))     # Fast Fourier propagator
Uz = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
Az = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
Uy = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
Ay = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
Ux = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
Ax = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)

# Uniform index for now TODO: Change this
n = n_h*np.ones((xy_cells,xy_cells,unique_layers),dtype=np.float_)
## For first two steps of Ex
Uz[:,:,0] = Ez
Az[:,:,0] = FFT2(Uz[:,:,0])

Uz[:,:,1] = Ez2
Az[:,:,1] = FFT2(Uz[:,:,1])

current_step = 2
Uz,Az, _, _ = propagate_Fourier(Uz, Az,FDFD_depth, current_step, dx, dz, xy_cells, n, imaging_depth_indices, absorption_padding, Absorption_strength, wavelength)

Uy[:,:,0] = Ey
Ay[:,:,0] = FFT2(Uz[:,:,0])

Uy[:,:,1] = Ey2
Ay[:,:,1] = FFT2(Uz[:,:,1])

current_step = 2
Uy,Ay, _, _ = propagate_Fourier(Uy, Ay,FDFD_depth, current_step, dx, dz, xy_cells, n, imaging_depth_indices, absorption_padding, Absorption_strength, wavelength)

Ux[:,:,0] = Ex
Ax[:,:,0] = FFT2(Uz[:,:,0])

Ux[:,:,1] = Ex2
Ax[:,:,1] = FFT2(Uz[:,:,1])

current_step = 2
Ux,Ax, _, _ = propagate_Fourier(Ux, Ax,FDFD_depth, current_step, dx, dz, xy_cells, n, imaging_depth_indices, absorption_padding, Absorption_strength, wavelength)


# Stuff at focus
Exf,Eyf,Ezf,_ = TightFocus(seed,dx_seed,wavelength,n_h,focus_depth,0,target_dx)

fig, ax = plt.subplots(4, 3)
axis = 10**6*dx_seed*indices
ax[0][0].pcolormesh(axis,axis,np.abs(seed)**2)
ax[0][0].title.set_text('Seed Intensity')

axis = 10**6*dx*indices
ax[0][1].pcolormesh(axis,axis,np.abs(Exf)**2+np.abs(Eyf)**2+np.abs(Ezf)**2)
ax[0][1].title.set_text('Intensity from Debye-Wolf calculation')
ax[0][2].pcolormesh(axis,axis,np.abs(Ux[:,:,1])**2+np.abs(Uy[:,:,1])**2+np.abs(Uz[:,:,1])**2)
ax[0][2].title.set_text('Intensity from DW + FD calculation')

ax[1][0].pcolormesh(axis,axis,np.abs(Exf))
ax[1][0].title.set_text("Debye-Wolf Ex")
ax[1][1].pcolormesh(axis,axis,np.abs(Eyf))
ax[1][1].title.set_text("Debye-Wolf Ey")
ax[1][2].pcolormesh(axis,axis,np.abs(Ezf))
ax[1][2].title.set_text("Debye-Wolf Ez")

ax[2][0].pcolormesh(axis,axis,np.abs(Ux[:,:,1]))
ax[2][0].title.set_text("DW + FD Ex")
ax[2][1].pcolormesh(axis,axis,np.abs(Uy[:,:,1]))
ax[2][1].title.set_text("DW + FD Ey")
ax[2][2].pcolormesh(axis,axis,np.abs(Uz[:,:,1]))
ax[2][2].title.set_text("DW + FD Ez")

ax[3][0].pcolormesh(axis,axis,np.abs(Ex))
ax[3][0].title.set_text("Hand off Ex")
ax[3][1].pcolormesh(axis,axis,np.abs(Ey))
ax[3][1].title.set_text("Hand off Ey")
ax[3][2].pcolormesh(axis,axis,np.abs(Ez))
ax[3][2].title.set_text("Hand off Ez")


print("--- %s seconds ---" % '%.2f'%(time.time() - start_time))
plt.show()
