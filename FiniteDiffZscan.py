# FD z-scan
import numpy as np
import time
import matplotlib.pyplot as plt
from SeedBeams import LG_OAM_beam, HG_beam, Gaussian_beam
import seaborn as sns
from DebyeWolfIntegral import TightFocus, SpotSizeCalculator
from FieldPlots import VortexNull
from PropagationAlgorithm import propagate_FiniteDifference
from FriendlyFourierTransform import FFT2, iFFT2
from scipy.interpolate import RegularGridInterpolator

start_time = time.time()

# Simulation parameters
wavelength = 500e-9
n_h = 1.33  # Homogenous part of refractive index
xy_cells = 1024    # Keep this a power of 2 for efficient FFT
unique_layers = 20
n = n_h*np.ones((xy_cells,xy_cells,unique_layers),dtype=np.float_)

start_dist = -10e-6
stop_dist = 10e-6
dz = 25e-9

beam_radius = 100e-6
focus_depth = 1e-3
FD_dist = stop_dist-start_dist
dx = dy = 8*beam_radius/(xy_cells)
expected_spot_size = SpotSizeCalculator(focus_depth,beam_radius,n_h,wavelength,np.abs(start_dist))
steps = int(FD_dist/dz)   # Make sure its a multiple of 4
if 2*beam_radius > 0.5*dx*xy_cells:
    ValueError("Beam is larger than simulation cross section")

if beam_radius<50*dx:
    raise ValueError('Sampling too coarse. Decrease dx')
    
absorption_padding = 3*dx # Thickness of absorbing boundary
Absorption_strength = 5

beam_type = 'G' # 'HG, 'LG', 'G'
l = 1  # Topological charge for LG beam
(u,v) = (1,0)   # Mode numbers for HG beam

if beam_type=='LG':
    seed = LG_OAM_beam(xy_cells, dx, beam_radius, l)
elif beam_type=='HG':
    seed = HG_beam(xy_cells, dx, beam_radius, u,v)
else:
    seed = Gaussian_beam(xy_cells, dx, beam_radius)

z_cross_section_profile_x = np.zeros((steps,xy_cells))
z_cross_section_profile_y = np.zeros((steps,xy_cells))
z_cross_section_profile_x_Fourier = np.zeros((100,xy_cells))
z_cross_section_profile_y_Fourier = np.zeros((100,xy_cells))

sns.heatmap(np.abs(seed))
plt.show()
Ex1,Ey1,Ez1,_ = TightFocus(seed,np.zeros(seed.shape),dx,wavelength,n_h,focus_depth,-start_dist,3*expected_spot_size/xy_cells)
Ex2,Ey2,Ez2,dx_TightFocus = TightFocus(seed,np.zeros(seed.shape),dx,wavelength,n_h,focus_depth,-start_dist-dz,3*expected_spot_size/xy_cells)
sns.heatmap(np.abs(Ex1))
plt.show()
dx = dx_TightFocus

U = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
U[:,:,0] = Ex1
U[:,:,1] = Ex2


k0 = 2*np.pi/wavelength
indices = np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
f = 1/(dx*xy_cells)*indices
fxfx,fyfy = np.meshgrid(f,f)
mask = ((fxfx**2+fyfy**2)<(1/wavelength)**2).astype(float)
dz_dx = dz/dx

for i in range(steps):
    d2Udx2 = (np.roll(U[:,:,1],1,axis=0)+np.roll(U[:,:,1],-1,axis=0)-2*U[:,:,1])
    d2Udy2 = (np.roll(U[:,:,1],1,axis=1)+np.roll(U[:,:,1],-1,axis=1)-2*U[:,:,1])
    U[:,:,2] = 2*U[:,:,1]-U[:,:,0]-(dz_dx**2)*(d2Udx2+d2Udy2)-(dz*k0*n[:,:,i%unique_layers])**2*U[:,:,1]
    U[:,:,2] =  iFFT2(mask*FFT2(U[:,:,2]))
    U[:,:,0] = U[:,:,1]
    U[:,:,1] = U[:,:,2]

    z_cross_section_profile_y[i,:] = (np.abs(U[:,:,2])**2)[:,int(xy_cells/2)]
    z_cross_section_profile_x[i,:] = (np.abs(U[:,:,2])**2)[int(xy_cells/2),:]

k = 2*np.pi*n_h/wavelength
dk = 2*np.pi/(dx*xy_cells)
kxkx, kyky = np.meshgrid(dk*indices,dk*indices)
z_scan_depths = FD_dist/100*np.linspace(0,99,100,dtype=np.int_)

for i in range(100):
    z = z_scan_depths[i]
    H = np.exp(1j*z*np.emath.sqrt((k)**2-kxkx**2-kyky**2))
    Ex = iFFT2(FFT2(Ex2)*H)        

    z_cross_section_profile_y_Fourier[i,:] = (np.abs(Ex)**2)[:,int(xy_cells/2)]
    z_cross_section_profile_x_Fourier[i,:] = (np.abs(Ex)**2)[int(xy_cells/2),:]

indices = np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
axis = 10**6*dx_TightFocus*indices
z_scan_depths = np.linspace(start_dist,stop_dist,steps)
plt.pcolormesh(axis,z_scan_depths,z_cross_section_profile_x)
plt.show()
#plt.pcolormesh(axis,z_scan_depths,z_cross_section_profile_y)
#plt.show()
z_scan_depths = np.linspace(start_dist,stop_dist,100)
plt.pcolormesh(axis,z_scan_depths,z_cross_section_profile_x_Fourier)
plt.show()
#plt.pcolormesh(axis,z_scan_depths,z_cross_section_profile_y_Fourier)
#plt.show()
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

'''
axis = 10**6*dx_TightFocus*indices
plt.pcolormesh(axis,axis,np.abs(Exf)**2)
plt.gca().set_aspect('equal')
plt.show()
VortexNull(np.abs(Exf)**2, dx_TightFocus, beam_type, cross_sections = 19, num_samples = 1000)
'''
print("--- %s seconds ---" % '%.2f'%(time.time() - start_time))