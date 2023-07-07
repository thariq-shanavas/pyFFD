import numpy as np
import time
import matplotlib.pyplot as plt
from SeedBeams import LG_OAM_beam, HG_beam, Gaussian_beam
import seaborn as sns
from DebyeWolfIntegral import TightFocus, SpotSizeCalculator
from FieldPlots import VortexNull

plt.rcParams['figure.dpi']= 100
plt.rcParams.update({'font.size': 4})
plt.rcParams['pcolor.shading'] = 'auto'

start_time = time.time()

# Simulation parameters
wavelength = 500e-9
n_h = 1.33  # Homogenous part of refractive index
xy_cells = 128    # Keep this a power of 2 for efficient FFT
padding = 4096
FDFD_depth = 5e-6
FDFD_dx = 80e-9
beam_radius = 1e-3
focus_depth = 2.5e-3

xy_cells = int(2**np.ceil(np.log2(SpotSizeCalculator(focus_depth,beam_radius,n_h,wavelength,np.max(FDFD_depth))*1.5/FDFD_dx)))
dx = dy = 2*2*beam_radius/(xy_cells)
print('Cell size is ' + str(xy_cells))
print('NA is '+str(n_h*(2*beam_radius/focus_depth)))
beam_type = 'LG' # 'HG, 'LG', 'G'
l = 1  # Topological charge for LG beam
(u,v) = (1,0)   # Mode numbers for HG beam


if beam_type=='LG':
    seed_x = LG_OAM_beam(xy_cells, dx, beam_radius, l)
    seed_y = np.zeros(seed_x.shape)
elif beam_type=='HG':
    seed_y = HG_beam(xy_cells, dx, beam_radius, u,v)
    seed_x = np.zeros(seed_y.shape)
else:
    seed_y = Gaussian_beam(xy_cells, dx, beam_radius)
    seed_x = np.zeros(np.shape(seed_y))

Ex,Ey,Ez,dx_TightFocus = TightFocus(seed_x,seed_y,dx,wavelength,n_h,focus_depth,FDFD_depth,FDFD_dx,padding)


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


Focus_Intensity = np.abs(Ex)**2+np.abs(Ey)**2+np.abs(Ez)**2

plt.pcolormesh(axis,axis,np.abs(Focus_Intensity))
plt.show()

print("--- %s seconds ---" % '%.2f'%(time.time() - start_time))