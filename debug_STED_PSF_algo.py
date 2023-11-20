import numpy as np
import matplotlib.pyplot as plt
from helper_classes import Results_class
from SeedBeams import Gaussian_beam
from ParallelTightFocusContrastVsDepth import Tightfocus_LG
from DebyeWolfIntegral import SpotSizeCalculator
from SeedBeams import LG_OAM_beam

LG = np.load('Results/Contrast_LG.npy', allow_pickle=True)
saturation_factor = 5


xy_cells = LG[0].intensity_profiles[0].shape[0]
dx = LG[0].dx_for_export
excitation_spot_size = SpotSizeCalculator(LG[0].focus_depth,LG[0].beam_radius,LG[0].n_h,LG[0].wavelength,0)
ideal_donut_LG = LG_OAM_beam(xy_cells, dx, excitation_spot_size/2, 1)**2
I_sat = 1/saturation_factor*np.max(ideal_donut_LG)

excitationBeam = (Gaussian_beam(xy_cells,dx,excitation_spot_size/2))**2
depletionBeam = LG[0].intensity_profiles[3]   # 20um

eta = np.exp(-np.log(2)*depletionBeam/I_sat)
STED_psf = excitationBeam*eta
plt.subplot(221)
plt.pcolormesh(np.abs(ideal_donut_LG))


plt.subplot(222)
plt.pcolormesh(np.abs(excitationBeam))

plt.subplot(223)
plt.pcolormesh(np.abs(depletionBeam))

plt.subplot(224)
plt.pcolormesh(np.abs(STED_psf))