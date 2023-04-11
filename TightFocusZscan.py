# Debye-Wolf z-scan
import numpy as np
import time
import matplotlib.pyplot as plt
from SeedBeams import LG_OAM_beam, HG_beam, Gaussian_beam
#import seaborn as sns
from DebyeWolfIntegral import TightFocus
from FieldPlots import VortexNull


start_time = time.time()

# Simulation parameters
wavelength = 500e-9
dz = 50e-9
n_h = 1  # Homogenous part of refractive index
xy_cells = 512    # Keep this a power of 2 for efficient FFT

beam_radius = 100e-6
focus_depth = 1e-3
dx = dy = 10*2*beam_radius/(xy_cells)

if 2*beam_radius > 0.5*dx*xy_cells:
    # Beam diameter greater than half the length of the simulation cross section.
    ValueError("Beam is larger than simulation cross section")

beam_type = 'G' # 'HG, 'LG', 'G'
l = 1  # Topological charge for LG beam
(u,v) = (1,0)   # Mode numbers for HG beam

if beam_type=='LG':
    seed = LG_OAM_beam(xy_cells, dx, beam_radius, l)
elif beam_type=='HG':
    seed = HG_beam(xy_cells, dx, beam_radius, u,v)
else:
    seed = Gaussian_beam(xy_cells, dx, beam_radius)

z_scan_depths = 60e-8*np.linspace(-50,49,100,dtype=np.int_)
z_cross_section_profile_x = np.zeros((100,xy_cells))
z_cross_section_profile_y = np.zeros((100,xy_cells))

for i in range(100):
    Ex,Ey,Ez,dx_TightFocus = TightFocus(seed,dx,wavelength,n_h,focus_depth,z_scan_depths[i],0.05)
    z_cross_section_profile_y[i,:] = (np.abs(Ex)**2+np.abs(Ey)**2+np.abs(Ez)**2)[:,int(xy_cells/2)]
    z_cross_section_profile_x[i,:] = (np.abs(Ex)**2+np.abs(Ey)**2+np.abs(Ez)**2)[int(xy_cells/2),:]

indices = np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
axis = 10**6*dx_TightFocus*indices
plt.pcolormesh(axis,z_scan_depths,z_cross_section_profile_x)
plt.show()
plt.pcolormesh(axis,z_scan_depths,z_cross_section_profile_y)
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
Ex,Ey,Ez,dx_TightFocus = TightFocus(seed,dx,wavelength,n_h,focus_depth,0,0.8/2.66)
axis = 10**6*dx_TightFocus*indices
plt.pcolormesh(axis,axis,np.abs(Ex)**2+np.abs(Ey)**2+np.abs(Ez)**2)
plt.gca().set_aspect('equal')
plt.show()
VortexNull(np.abs(Ex)**2+np.abs(Ey)**2+np.abs(Ez)**2, dx_TightFocus, beam_type, cross_sections = 19, num_samples = 1000)

print("--- %s seconds ---" % '%.2f'%(time.time() - start_time))