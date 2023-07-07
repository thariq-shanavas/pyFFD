import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import hmean

class Results:
    # Saves results for all depths, for one instantiation of the tissue.
    def __init__(self, contrasts, contrast_std_deviations):
        self.depths = depths
        self.contrasts = contrasts
        self.contrast_std_deviations = contrast_std_deviations
        self.wavelength = wavelength
        self.FDFD_dx = FDFD_dx
        self.dz = dz
        self.beam_radius = beam_radius
        self.focus_depth = focus_depth
        self.unique_layers = unique_layers
        self.n_h = n_h
        self.ls = ls
        self.g = g
        self.xy_cells = xy_cells

LG = np.load('Results/Contrast_LG.npy', allow_pickle=True)
HG = np.load('Results/Contrast_HG.npy', allow_pickle=True)

contrast_vs_depth_LG = np.zeros((len(LG[0].depths),len(LG)))
contrast_vs_depth_HG = np.zeros((len(HG[0].depths),len(HG)))
for i in range(len(LG)):
    contrast_vs_depth_LG[:,i] = LG[i].contrasts
    contrast_vs_depth_HG[:,i] = HG[i].contrasts

plt.plot(LG[0].depths,np.median(contrast_vs_depth_LG,axis=1),label='LG')
plt.plot(HG[0].depths,np.median(contrast_vs_depth_HG,axis=1),label='HG')
plt.legend()