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

saturation_factor = 30
ls = 59     # Scattering mean free path, um
excitation_wavelength = 635e-9

LG = np.load('Results/Contrast_LG.npy', allow_pickle=True)
HG = np.load('Results/Contrast_HG.npy', allow_pickle=True)
num_runs = len(LG)

LG_extended = np.load('Results/Contrast_LG_extended.npy', allow_pickle=True)
HG_extended = np.load('Results/Contrast_HG_extended.npy', allow_pickle=True)

depletion_wavelength = LG[0].wavelength
dx = LG[0].dx_for_export
xy_cells = LG[0].intensity_profiles[0].shape[0]
depths = np.concatenate((LG[0].depths * 10**6,LG_extended[0].depths * 10**6))
#depths = LG[0].depths * 10**6
excitation_spot_size = SpotSizeCalculator(LG[0].focus_depth,LG[0].beam_radius,LG[0].n_h,excitation_wavelength,0) # 1/e^2 diameter of a gaussian at focus
excitationBeam = (Gaussian_beam(xy_cells,dx,excitation_spot_size/2))**2
depletion_spot_size = SpotSizeCalculator(LG[0].focus_depth,LG[0].beam_radius,LG[0].n_h,depletion_wavelength,0) # 1/e^2 diameter of a gaussian at focus
ideal_donut_LG = np.abs(LG_OAM_beam(xy_cells, dx, depletion_spot_size/2, 1))**2
ideal_donut_LG = ideal_donut_LG/(np.sum(ideal_donut_LG)*dx**2)
#ideal_donut_HG = ideal_donut_HG/(np.sum(ideal_donut_HG)*dx**2)

#I_sat_LG = 1/saturation_factor*np.max(ideal_donut_LG)
#I_sat_HG = 1/saturation_factor*np.max(ideal_donut_HG)
I_sat = 1/saturation_factor*np.max(ideal_donut_LG)

PSF_vs_depth_LG = np.zeros((len(LG[0].depths)+len(LG_extended[0].depths),num_runs))
#PSF_vs_depth_LG = np.zeros((len(LG[0].depths),num_runs))
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
            #field_profile_LG = field_profile_LG/(np.sum(field_profile_LG)*dx**2)
            PSF_vs_depth_LG[depth_index,run_number], PSF_centroid_LG[depth_index,run_number], STED_power_LG[depth_index,run_number] = STED_psf_fwhm(dx,attenuation_factor*excitationBeam,field_profile_LG, I_sat)
            
            field_profile_HG = HG[run_number].intensity_profiles[depth_index]
            #field_profile_HG = field_profile_HG/(np.sum(field_profile_HG)*dx**2)
            PSF_vs_depth_HG[depth_index,run_number], PSF_centroid_HG[depth_index,run_number], STED_power_HG[depth_index,run_number] = STED_psf_fwhm(dx,attenuation_factor*excitationBeam,field_profile_HG, I_sat)

        else:
            field_profile_LG = LG_extended[run_number].intensity_profiles[depth_index- len(LG[0].depths)]
            #field_profile_LG = field_profile_LG/(np.sum(field_profile_LG)*dx**2)
            PSF_vs_depth_LG[depth_index,run_number], PSF_centroid_LG[depth_index,run_number], STED_power_LG[depth_index,run_number] = STED_psf_fwhm(dx,attenuation_factor*excitationBeam,field_profile_LG, I_sat)
            
            field_profile_HG = HG_extended[run_number].intensity_profiles[depth_index - len(LG[0].depths)]
            #field_profile_HG = field_profile_HG/(np.sum(field_profile_HG)*dx**2)
            PSF_vs_depth_HG[depth_index,run_number], PSF_centroid_HG[depth_index,run_number], STED_power_HG[depth_index,run_number] = STED_psf_fwhm(dx,attenuation_factor*excitationBeam,field_profile_HG, I_sat)


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
LG_deviation = np.median(PSF_centroid_LG,axis=1)
HG_deviation = np.median(PSF_centroid_HG,axis=1)
power_LG = np.mean(STED_power_LG,axis=1)
power_HG = np.mean(STED_power_HG,axis=1)

# Ideal case at surface
# Note: Does not use Debye-Wolf integral. There might be a very small error.

'''
ideal_PSF_diameter, _, ideal_STED_power = STED_psf_fwhm(dx,excitationBeam,2*ideal_donut_LG, I_sat)
depths[0] = 0
LG_figure_of_merit[0] = ideal_PSF_diameter
HG_figure_of_merit[0] = ideal_PSF_diameter
LG_deviation[0] = 0
HG_deviation[0] = 0
power_LG[0] = ideal_STED_power
power_HG[0] = ideal_STED_power
depths = np.concatenate(([0],depths))
LG_figure_of_merit = np.concatenate(([ideal_PSF_diameter],LG_figure_of_merit))
HG_figure_of_merit = np.concatenate(([ideal_PSF_diameter],HG_figure_of_merit))
LG_deviation = np.concatenate(([0],LG_deviation))
HG_deviation = np.concatenate(([0],HG_deviation))
power_LG = np.concatenate(([ideal_STED_power],power_LG))
power_HG = np.concatenate(([ideal_STED_power],power_HG))
'''

plt.rcParams.update({'font.size': 20})
plt.rcParams['pcolor.shading'] = 'auto'
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 2

plt.figure(figsize=[16,4],dpi=500)
plt.subplot(1,3,1)
plt.plot(depths,LG_figure_of_merit,label='LG', marker = 'o')
plt.plot(depths,HG_figure_of_merit,label='HG', marker = 'o')
plt.plot([0,25,50,75],[90,100,140,165],label='LG, Expt.',linestyle='None',marker = 'D')
plt.legend(loc='lower left')
plt.xlabel('Tissue depth (μm)')
plt.ylabel('FWHM (nm)')
plt.ylim(bottom = 0)

plt.subplot(1,3,2)
plt.plot(depths,LG_deviation,label='LG', marker = 'o')
plt.plot(depths,HG_deviation,label='HG', marker = 'o')
plt.legend(loc='lower left')
plt.ylim(bottom = 0)
plt.xlabel('Tissue depth (μm)')
plt.ylabel('PSF deviation (nm)')

plt.subplot(1,3,3)
plt.plot(depths,100/8*np.exp(-depths/ls)*power_LG,label='LG', marker = 'o')
plt.plot(depths,100/8*np.exp(-depths/ls)*power_HG,label='HG', marker = 'o')
plt.legend(loc='lower left')
plt.ylim(bottom = 0)
plt.xlabel('Tissue depth (μm)')
plt.ylabel('STED Power (a.u.)')
plt.tight_layout()
plt.show()