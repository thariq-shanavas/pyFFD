import numpy as np
import matplotlib.pyplot as plt
from helper_classes import Results_class
from SeedBeams import Gaussian_beam
from DebyeWolfIntegral import SpotSizeCalculator
from SeedBeams import LG_OAM_beam
from BeamQuality import STED_psf_fwhm

LG = np.load('Results/Contrast_HG.npy', allow_pickle=True)

depletionBeam = LG[0].intensity_profiles[0]   # 20um
saturation_factor = 20


xy_cells = LG[0].intensity_profiles[0].shape[0]
dx = LG[0].dx_for_export
excitation_spot_size = SpotSizeCalculator(LG[0].focus_depth,LG[0].beam_radius,LG[0].n_h,LG[0].wavelength,0)
ideal_donut_LG = LG_OAM_beam(xy_cells, dx, excitation_spot_size/2, 1)**2
I_sat = 1/saturation_factor*np.max(ideal_donut_LG)

excitationBeam = (Gaussian_beam(xy_cells,dx,excitation_spot_size/2))**2


eta = np.exp(-np.log(2)*depletionBeam/I_sat)
STED_psf = excitationBeam*eta
plt.figure(figsize=[24,7])

plt.subplot(131)
plt.pcolormesh(np.abs(excitationBeam[250:750,250:750]))
plt.gca().set_aspect('equal')

plt.subplot(132)
plt.pcolormesh(np.abs(depletionBeam[250:750,250:750]))
plt.gca().set_aspect('equal')

plt.subplot(133)
plt.pcolormesh(np.abs(STED_psf[250:750,250:750]))
plt.gca().set_aspect('equal')
plt.show()
print(STED_psf_fwhm(dx,excitationBeam,depletionBeam, I_sat))