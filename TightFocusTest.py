import numpy as np
import time
import matplotlib.pyplot as plt
from SeedBeams import LG_OAM_beam, HG_beam, Gaussian_beam
import seaborn as sns
from DebyeWolfIntegral import TightFocus
from FieldPlots import VortexNull

plt.rcParams['figure.dpi']= 300
plt.rcParams.update({'font.size': 4})
plt.rcParams['pcolor.shading'] = 'auto'

start_time = time.time()

# Simulation parameters
wavelength = 500e-9
n_h = 1  # Homogenous part of refractive index
xy_cells = 512    # Keep this a power of 2 for efficient FFT

beam_radius = 1e-3
focus_depth = 5e-3
dx = dy = 5e-3/(xy_cells)

if 2*beam_radius > dx*xy_cells:
    # Beam diameter greater than half the length of the simulation cross section.
    raise ValueError("Beam is larger than simulation cross section")

if beam_radius<50*dx:
    raise ValueError('Sampling too coarse. Decrease dx')

beam_type = 'G' # 'HG, 'LG', 'G'
l = 1  # Topological charge for LG beam
(u,v) = (1,0)   # Mode numbers for HG beam

if beam_type=='LG':
    seed = LG_OAM_beam(xy_cells, dx, beam_radius, l)
elif beam_type=='HG':
    seed_y = HG_beam(xy_cells, dx, beam_radius, u,v)
    seed_x = np.zeros(seed_y.shape)
else:
    seed_y = Gaussian_beam(xy_cells, dx, beam_radius)
    seed_x = np.zeros(np.shape(seed_y))

Ex,Ey,Ez,dx_TightFocus = TightFocus(seed_x,seed_y,dx,wavelength,n_h,focus_depth,MeasurementPlane_z=1e-6,target_dx=10e-9)


indices = np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
# xx, yy = np.meshgrid(dx*indices,dx*indices)

fig, ax = plt.subplots(2, 3)
axis = 10**6*dx*indices
ax[0][0].pcolormesh(axis,axis,+np.abs(seed_x)**2+np.abs(seed_y)**2)
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
plt.show()

if beam_type=='HG':
    Focus_Intensity = np.abs(Ex)**2+np.abs(Ey)**2+np.abs(Ez)**2
    Focus_Intensity = (Focus_Intensity+np.transpose(Focus_Intensity))/2
else:
    Focus_Intensity = np.abs(Ex)**2+np.abs(Ey)**2+np.abs(Ez)**2
VNull = VortexNull(Focus_Intensity, dx, beam_type, cross_sections = 19, num_samples = 1000)

plt.pcolormesh(axis,axis,np.abs(Focus_Intensity))
plt.show()

print("--- %s seconds ---" % '%.2f'%(time.time() - start_time))