import numpy as np
import time
import matplotlib.pyplot as plt
from SeedBeams import LG_OAM_beam, HG_beam, Gaussian_beam
#import seaborn as sns
from DebyeWolfIntegral import TightFocus
from FriendlyFourierTransform import FFT2, iFFT2
from scipy.interpolate import RegularGridInterpolator


start_time = time.time()

# Simulation parameters
wavelength = 500e-9
dz = 50e-9
n_h = 1.518  # Homogenous part of refractive index
xy_cells = 1024    # Keep this a power of 2 for efficient FFT

beam_radius = 100e-6
focus_depth = 1e-3
dx = dy = 10*2*beam_radius/(xy_cells)

if 2*beam_radius > 0.5*dx*xy_cells:
    # Beam diameter greater than half the length of the simulation cross section.
    ValueError("Beam is larger than simulation cross section")

beam_type = 'LG' # 'HG, 'LG', 'G'
l = 1  # Topological charge for LG beam
(u,v) = (1,0)   # Mode numbers for HG beam

if beam_type=='LG':
    seed = LG_OAM_beam(xy_cells, dx, beam_radius, l)
elif beam_type=='HG':
    seed = HG_beam(xy_cells, dx, beam_radius, u,v)
else:
    seed = Gaussian_beam(xy_cells, dx, beam_radius)

z_scan_depths = 30e-8*np.linspace(-50,49,100,dtype=np.int_)
z_cross_section_profile_x = np.zeros((100,xy_cells))
z_cross_section_profile_y = np.zeros((100,xy_cells))

dk = 2*np.pi/(dx*xy_cells)
indices = np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
kxkx, kyky = np.meshgrid(dk*indices,dk*indices)
f = 1/(dx*xy_cells)*indices
k = 2*np.pi*n_h/wavelength
k0 = 2*np.pi/wavelength

steps = 100
dx_new = 1.5e-6/xy_cells

for i in range(100):
    z = focus_depth - z_scan_depths[i]
    H = np.exp(1j*z*np.emath.sqrt((k)**2-kxkx**2-kyky**2))
    Ex = iFFT2(FFT2(seed)*H)        
    interpEx = RegularGridInterpolator((dx*indices,dx*indices), Ex, bounds_error = False, fill_value = 0)
    xx_new, yy_new = np.meshgrid(dx_new*indices,dx_new*indices,indexing='ij')
    Ex = interpEx((xx_new,yy_new))
    
    z_cross_section_profile_y[i,:] = (np.abs(Ex)**2)[:,512]
    z_cross_section_profile_x[i,:] = (np.abs(Ex)**2)[512,:]


axis = 10**6*dx_new*indices
plt.pcolormesh(axis,10**6*z_scan_depths,z_cross_section_profile_x)
plt.show()
plt.pcolormesh(axis,10**6*z_scan_depths,z_cross_section_profile_y)
'''
indices = np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
# xx, yy = np.meshgrid(dx*indices,dx*indices)

fig, ax = plt.subplots(2, 3)
axis = 10**6*dx*indices
ax[0][0].pcolormesh(axis,axis,np.abs(seed)**2)
ax[0][0].title.set_text('Seed Intensity')

axis = 10**6*dx_TightFocus*indices
ax[0][1].pcolormesh(axis,axis,np.abs(Ex)**2+np.abs(Ey)**2+np.abs(Ez)**2)
ax[0][1].title.set_text('Intensity at focus')
ax[1][0].pcolormesh(axis,axis,np.abs(Ex))
ax[1][0].title.set_text("Ex")
ax[1][1].pcolormesh(axis,axis,np.abs(Ey))
ax[1][1].title.set_text("Ey")
ax[1][2].pcolormesh(axis,axis,np.abs(Ez))
ax[1][2].title.set_text("Ez")
'''
plt.show()
z = focus_depth
H = np.exp(1j*z*np.emath.sqrt((k)**2-kxkx**2-kyky**2))
Ex = iFFT2(FFT2(seed)*H)        
interpEx = RegularGridInterpolator((dx*indices,dx*indices), Ex, bounds_error = False, fill_value = 0)
xx_new, yy_new = np.meshgrid(dx_new*indices,dx_new*indices,indexing='ij')
Ex = interpEx((xx_new,yy_new))
plt.pcolormesh(axis,axis,np.abs(Ex)**2)
plt.gca().set_aspect('equal')
plt.show()

print("--- %s seconds ---" % '%.2f'%(time.time() - start_time))