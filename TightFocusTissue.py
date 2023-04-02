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

# Simulation parameters
wavelength = 500e-9
dz = 50e-9
dx = dy = 2*wavelength # Minimum resolution = lambda/(n*sqrt(2)) for finite difference. Any lower and the algorithm is numerically unstable
n_h = 1.33  # Homogenous part of refractive index
xy_cells = 1024    # Keep this a power of 2 for efficient FFT

beam_radius = 15e-6
focus_depth = 30e-6

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

Ex,Ey,Ez,dx_TightFocus = TightFocus(seed,dx,wavelength,n_h,focus_depth,0)
