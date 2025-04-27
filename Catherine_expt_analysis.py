import numpy as np
import matplotlib.pyplot as plt
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

saturation_factor = 19
ls = 96     # Scattering mean free path, um
excitation_wavelength = 910e-9

LG = np.load('Results/Catherine_expt.npy', allow_pickle=True)
#HG = np.load('Results/HG_result.npy', allow_pickle=True)
num_runs = len(LG)

depletion_wavelength = LG[0].wavelength
dx = LG[0].dx_for_export
xy_cells = LG[0].intensity_profiles[0].shape[0]
depths = LG[0].depths * 10**6
excitation_spot_size = SpotSizeCalculator(LG[0].focus_depth,LG[0].beam_radius,LG[0].n_h,excitation_wavelength,0) # 1/e^2 diameter of a gaussian at focus
excitationBeam = (Gaussian_beam(xy_cells,dx,excitation_spot_size/2))**2
depletion_spot_size = SpotSizeCalculator(LG[0].focus_depth,LG[0].beam_radius,LG[0].n_h,depletion_wavelength,0) # 1/e^2 diameter of a gaussian at focus

# Factor of 2 here to account for adding y-polarization in the simulation.
ideal_donut_LG = 2*np.abs(LG_OAM_beam(xy_cells, dx, depletion_spot_size/2, 1))**2
#ideal_donut_HG = ideal_donut_HG/(np.sum(ideal_donut_HG)*dx**2)

#I_sat_LG = 1/saturation_factor*np.max(ideal_donut_LG)
#I_sat_HG = 1/saturation_factor*np.max(ideal_donut_HG)
I_sat = 1/saturation_factor*np.max(ideal_donut_LG)

PSF_vs_depth_LG = np.zeros((len(LG[0].depths),num_runs))
PSF_centroid_LG = np.zeros(PSF_vs_depth_LG.shape)

PSF_vs_depth_HG = np.zeros(PSF_vs_depth_LG.shape)
PSF_centroid_HG = np.zeros(PSF_vs_depth_LG.shape)

STED_power_LG = np.zeros(PSF_vs_depth_LG.shape)
STED_power_HG = np.zeros(PSF_vs_depth_LG.shape)

for run_number in range(num_runs):
    for depth_index in range(len(depths)):
        # Note: The absorption mean free path is vastly bigger than scattering mean free path
        # The scattering angles are also small in tissue, so photons are unlikely to be scattered out of the simulation volume.
        # Therefore, I'm assuming that the total optical power in the finite difference simulation volume is conserved for every transverse cross section. This should be true to within a small margin of error
        # However, I find that the total power calculated at the focual plane is sometimes a little larger than what was sent in. This is most likely due to errors from discretization.
        # So I'm normalizing power at focus before calculating the STED FWHM

        attenuation_factor = np.exp(-depths[depth_index]/ls)

        if depth_index<len(LG[0].depths):

            field_profile_LG = LG[run_number].intensity_profiles[depth_index]
            PSF_vs_depth_LG[depth_index,run_number], PSF_centroid_LG[depth_index,run_number], STED_power_LG[depth_index,run_number] = STED_psf_fwhm(dx,attenuation_factor*excitationBeam,field_profile_LG, I_sat)
            
            #field_profile_HG = HG[run_number].intensity_profiles[depth_index]
            #PSF_vs_depth_HG[depth_index,run_number], PSF_centroid_HG[depth_index,run_number], STED_power_HG[depth_index,run_number] = STED_psf_fwhm(dx,attenuation_factor*excitationBeam,field_profile_HG, I_sat)

        '''
        plt.pcolormesh(excitationBeam>fluorescenceThreshold)
        plt.show()
        plt.pcolormesh(field_profile_LG<I_sat)
        plt.show()
        plt.pcolormesh(np.logical_and((excitationBeam>fluorescenceThreshold),(field_profile_LG<I_sat)))
        plt.show()
        '''

print("Adding 100 nm to adjust for size of gold sphere.")
LG_figure_of_merit = np.median(PSF_vs_depth_LG,axis=1)+100
#HG_figure_of_merit = np.mean(PSF_vs_depth_HG,axis=1)
LG_deviation = np.median(PSF_centroid_LG,axis=1)
#HG_deviation = np.median(PSF_centroid_HG,axis=1)
power_LG = np.mean(STED_power_LG,axis=1)
#power_HG = np.mean(STED_power_HG,axis=1)

#####
plt.rcParams.update({'font.size': 18})
plt.rcParams['pcolor.shading'] = 'auto'
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 2

plt.figure(figsize=[12,4],dpi=500)
plt.subplot(1,2,1)
plt.plot(depths,LG_figure_of_merit,label='Simulation', marker = 'o')
#plt.plot(depths,HG_figure_of_merit,label='HG, Sim.', marker = 'D')
plt.plot([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],[186, 184, 230, 236, 270, 247, 348, 283, 266, 271, 297],label='Experiment',linestyle='None',marker = 's')
plt.legend(loc='lower right')
plt.xlabel('Tissue depth (μm)')
plt.ylabel('FWHM (nm)')
plt.ylim([0,400])

plt.subplot(1,2,2)
plt.plot(depths,30+10*np.log10(np.exp(-depths/ls)*power_LG),label='Simulation', marker = 'o')
#plt.plot(depths,30+10*np.log10(np.exp(-depths/ls)*power_HG),label='HG, Sim.', marker = 'D')
plt.legend(loc='lower left')
#plt.ylim(bottom = 0)
plt.xlabel('Tissue depth (μm)')
plt.ylabel('SNR (dB)')
plt.tight_layout()
plt.show()