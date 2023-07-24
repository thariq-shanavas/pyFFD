import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import hmean
from ParallelTightFocusContrastVsDepth import Results

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