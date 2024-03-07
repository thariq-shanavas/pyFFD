# This is a very minimal example
# We simulate a gaussian beam focused into tissue using a high-NA objective

import numpy as np
from PropagationAlgorithm import Propagate
from SeedBeams import Gaussian_beam
from DebyeWolfIntegral import TightFocus, SpotSizeCalculator
from FriendlyFourierTransform import optimal_cell_size
from GenerateRandomTissue import RandomTissue
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

FDFD_dz = 25e-9         # step size in the direction of beam propagation. Keep this at about lambda/20
FDFD_dx = 50e-9
wavelength = 500e-9
ls = 59e-6  # Mean free path in tissue
g = 0.90    # Anisotropy factor

# These three parameters determine the NA of the system
beam_radius = 1e-3
focus_depth = 2.5e-3    # Depth at which the beam is focused. Note that this is the focal length in the medium.
n_h = 1.33  # Homogenous part of tissue refractive index, also the index of the immersion medium

depth = 10e-6 # Focus light 10 microns under tissue
suppress_evanescent = True  # Applies the freq. domain filter explained in supplementary material

unique_layers = 110
# Procedurally generate these many unique layers of tissue. 
# If unique_layers*FDFD_dz is less than the tissue depth, the layers are re-used cyclically.

min_xy_cells = 255      # Minimum cells to be used. This is important because about 10 cells on the outer edge of the simulation form an absorbing boundary.
# xy_cells is the number of discretization points.
# i.e., the lateral cross section is represented by an xy_cells by xy_cells matrix

# Figure out the lateral discretization of space, i.e., dx
spot_size = SpotSizeCalculator(focus_depth,beam_radius,n_h,wavelength,depth)    # Calculate the spot size at the tissue surface
xy_cells = optimal_cell_size(spot_size, FDFD_dx, min_xy_cells)      # optimal_cell_size() returns the size of the matrix that is fastest for FFT, but also big enough that the beam is not close to the edge

# Generate the seed beam. Use the same matrix size as before, but choose a convenient dx for the seed
# In this example, we use a linearly polarized gaussian beam
seed_dx = 5*beam_radius/(xy_cells)
seed_x = Gaussian_beam(xy_cells,seed_dx,beam_radius)
seed_y = np.zeros_like(seed_x)

# Use Debye-Wolf integral to calculate field at two planes near the surface of the tissue.
Ex,Ey,Ez,_ = TightFocus(seed_x,seed_y,seed_dx,wavelength,n_h,focus_depth,depth,FDFD_dx,2048)
Ex2,Ey2,Ez2,_ = TightFocus(seed_x,seed_y,seed_dx,wavelength,n_h,focus_depth,depth-FDFD_dz,FDFD_dx,2048)

# The finite difference solver expects the initial conditions to be saved in NxNx3 matrices for each polarization.
# i.e., 
# Ux[:,:,0], Ux[:,:,1] and Ux[:,:,2] are the Ex fields in three transverse planes, separated by FDFD_dz
# The initial conditions are provided in Ux[:,:,0] and Ux[:,:,1]

Uz = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
Uy = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)
Ux = np.zeros((xy_cells,xy_cells,3),dtype=np.complex64)

[Ux[:,:,0], Uy[:,:,0], Uz[:,:,0]] = [Ex, Ey, Ez]
[Ux[:,:,1], Uy[:,:,1], Uz[:,:,1]] = [Ex2, Ey2, Ez2]


# Generate tissue model
n = RandomTissue([xy_cells, wavelength, FDFD_dx, FDFD_dz, n_h, ls, g, unique_layers, 0])
# Use the finite difference solver to propagate fields to the focal plane.
# Propagate() is the basic finite difference solver
# Propagate_adaptiveResolution() avoids simulating tissue without any light as detailed in the supplementary material.

Ux,Uy,Uz = Propagate(Ux, Uy, Uz, depth, FDFD_dx, FDFD_dz, xy_cells, n, wavelength, suppress_evanescent = True)

# Plot results
exportResolution = 1000     # Save fields as a resultSaveResolution x resultSaveResolution matrix. Keep this even.
dx_for_export = 6*SpotSizeCalculator(focus_depth,beam_radius,n_h,wavelength,0)/exportResolution
intensity = np.abs(Ux[:,:,0])**2+np.abs(Uy[:,:,0])**2+np.abs(Uz[:,:,0])**2
original_axis = 10**6*FDFD_dx*np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
plotting_axes = 10**6*dx_for_export*np.linspace(-exportResolution/2,exportResolution/2-1,exportResolution,dtype=np.int_)
xx_export, yy_export = np.meshgrid(plotting_axes,plotting_axes, indexing='ij')  # Mesh grid used for exporting data
plotting_field = RegularGridInterpolator((original_axis,original_axis),intensity, bounds_error = True, method='linear')((xx_export, yy_export))
fig = plt.figure()
plt.gca().set_aspect('equal')
plt.pcolormesh(plotting_axes,plotting_axes,RegularGridInterpolator((original_axis,original_axis),intensity, bounds_error = True, method='linear')((xx_export, yy_export)))
plt.title("Beam at focus", weight='bold')
plt.xlabel("x ($µm$)", weight='bold', fontsize=12)
plt.xticks(weight = 'bold', fontsize=12)
plt.ylabel("y ($µm$)", weight='bold', fontsize=12)
plt.yticks(weight = 'bold', fontsize=12)
plt.show()