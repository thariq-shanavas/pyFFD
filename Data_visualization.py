import numpy as np
import matplotlib.pyplot as plt
from ParallelTightFocusContrastVsDepth import Tightfocus_LG
from helper_classes import Results_class
from BeamQuality import STED_psf_fwhm
from DebyeWolfIntegral import SpotSizeCalculator
from SeedBeams import Gaussian_beam
from SeedBeams import LG_OAM_beam

# VERY IMPORTANT NOTE
# Do not rely on the order of items in contrasts or intensity_profile.
# Match the depth list object to any result list item.

# Parameters
# Saturation factor is the peak power of an ideal donut over the saturation power, for either type.
# To simulate increasing donut power, increase this factor.

saturation_factor = 50

LG = np.load('Results/Contrast_LG.npy', allow_pickle=True)
HG = np.load('Results/Contrast_HG.npy', allow_pickle=True)
num_runs = len(LG)


dx = LG[0].dx_for_export
xy_cells = LG[0].intensity_profiles[0].shape[0]
depths = LG[0].depths * 10**6
excitation_spot_size = SpotSizeCalculator(LG[0].focus_depth,LG[0].beam_radius,LG[0].n_h,LG[0].wavelength,0) # 1/e^2 diameter of a gaussian at focus
excitationBeam = (Gaussian_beam(xy_cells,dx,excitation_spot_size/2))**2
ideal_donut_LG = np.abs(LG_OAM_beam(xy_cells, dx, excitation_spot_size/2, 1))**2
I_sat = 1/saturation_factor*np.max(ideal_donut_LG)
#_,_, ideal_donut_HG = Tightfocus_HG([0,'',0])

# Normalizing power to unity
ideal_donut_LG = ideal_donut_LG/(np.sum(ideal_donut_LG)*dx**2)
#ideal_donut_HG = ideal_donut_HG/(np.sum(ideal_donut_HG)*dx**2)

#I_sat_LG = 1/saturation_factor*np.max(ideal_donut_LG)
#I_sat_HG = 1/saturation_factor*np.max(ideal_donut_HG)
I_sat = 1/saturation_factor*np.max(ideal_donut_LG)

PSF_vs_depth_LG = np.zeros((len(LG[0].depths),num_runs))
PSF_centroid_deviation_LG = np.zeros((len(LG[0].depths),num_runs))
donuteBeamPowerLG = np.zeros((len(LG[0].depths),num_runs))
PSF_vs_depth_HG = np.zeros(PSF_vs_depth_LG.shape)
PSF_centroid_deviation_HG = np.zeros(PSF_centroid_deviation_LG.shape)
donuteBeamPowerHG = np.zeros(donuteBeamPowerLG.shape)

for run_number in range(num_runs):
    for depth_index in range(len(depths)):
        # Note: The absorption mean free path is vastly bigger than scattering mean free path
        # The scattering angles are also small in tissue, so photons are unlikely to be scattered out of the simulation volume.
        # Therefore, I'm assuming that the total optical power in the finite difference simulation volume is conserved for every transverse cross section. This should be true to within a small margin of error
        # However, I find that the total power calculated at the focual plane is sometimes a little larger than what was sent in. This is most likely due to errors from discretization.
        # So I'm normalizing power at focus before calculating the STED FWHM

        field_profile_LG = LG[run_number].intensity_profiles[depth_index]
        #field_profile_LG = field_profile_LG/(np.sum(field_profile_LG)*dx**2)
        donuteBeamPowerLG[depth_index,run_number] = np.sum(field_profile_LG*dx**2)
        PSF_vs_depth_LG[depth_index,run_number], PSF_centroid_deviation_LG[depth_index,run_number] = STED_psf_fwhm(dx,excitationBeam,field_profile_LG, I_sat)
        
        field_profile_HG = HG[run_number].intensity_profiles[depth_index]
        #field_profile_HG = field_profile_HG/(np.sum(field_profile_HG)*dx**2)
        donuteBeamPowerHG[depth_index,run_number] = np.sum(field_profile_HG*dx**2)
        PSF_vs_depth_HG[depth_index,run_number], PSF_centroid_deviation_HG[depth_index,run_number] = STED_psf_fwhm(dx,excitationBeam,field_profile_HG, I_sat)
        '''
        plt.pcolormesh(excitationBeam>fluorescenceThreshold)
        plt.show()
        plt.pcolormesh(field_profile_LG<I_sat)
        plt.show()
        plt.pcolormesh(np.logical_and((excitationBeam>fluorescenceThreshold),(field_profile_LG<I_sat)))
        plt.show()
        '''
LG_figure_of_merit = np.mean(PSF_vs_depth_LG,axis=1)
HG_figure_of_merit = np.mean(PSF_vs_depth_HG,axis=1)
plt.subplot(2,1,1)
plt.plot(depths,LG_figure_of_merit,label='LG', marker = 'o')
plt.plot(depths,HG_figure_of_merit,label='HG', marker = 'o')
plt.legend()

# plt.ylim(bottom = 0)
plt.gcf().set_dpi(300)
plt.rcParams.update({'font.size': 12})
plt.rcParams['pcolor.shading'] = 'auto'
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 2
plt.title("STED PSF for LG vs HG depletion beam", weight='bold')
plt.xlabel("Tissue depth ($µm$)", weight='bold', fontsize=12)
plt.xticks(weight = 'bold', fontsize=12)
plt.ylabel("FWHM (nm)", weight='bold', fontsize=12)
plt.yticks(weight = 'bold', fontsize=12)
plt.tight_layout()

plt.subplot(2,1,2)
plt.plot(depths,np.median(PSF_centroid_deviation_LG,axis=1),label='LG', marker = 'o')
plt.plot(depths,np.median(PSF_centroid_deviation_HG,axis=1),label='HG', marker = 'o')
plt.legend()

# plt.ylim(bottom = 0)
plt.xlabel("Tissue depth ($µm$)", weight='bold', fontsize=12)
plt.ylabel("Centroid deviation (nm)", weight='bold', fontsize=12)

plt.show()
plt.plot(depths,np.median(donuteBeamPowerLG,axis=1),label='LG', marker = 'o')
plt.plot(depths,np.median(donuteBeamPowerHG,axis=1),label='HG', marker = 'o')
plt.show()