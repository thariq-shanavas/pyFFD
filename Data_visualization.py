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

saturation_factor = 18
LG2_LG1_power_ratio = 10.8    # Use 10x as much power for LG-2 depletion beams
ls = 59     # Scattering mean free path, um
excitation_wavelength = 635e-9

LG = np.load('Results/Contrast_LG.npy', allow_pickle=True)
LG_2 = np.load('Results/Contrast_LG_2nd_order.npy', allow_pickle=True)
num_runs = len(LG)

depletion_wavelength = LG[0].wavelength
dx = LG[0].dx_for_export
xy_cells = LG[0].intensity_profiles[0].shape[0]

depths = LG[0].depths * 10**6
excitation_spot_size = SpotSizeCalculator(LG[0].focus_depth,LG[0].beam_radius,LG[0].n_h,excitation_wavelength,0) # 1/e^2 diameter of a gaussian at focus
excitationBeam = (Gaussian_beam(xy_cells,dx,excitation_spot_size/2))**2
depletion_spot_size = SpotSizeCalculator(LG[0].focus_depth,LG[0].beam_radius,LG[0].n_h,depletion_wavelength,0) # 1/e^2 diameter of a gaussian at focus
ideal_donut_LG = np.abs(LG_OAM_beam(xy_cells, dx, depletion_spot_size/2, 1))**2
ideal_donut_LG = ideal_donut_LG/(np.sum(ideal_donut_LG)*dx**2)
#ideal_donut_LG_2 = ideal_donut_LG_2/(np.sum(ideal_donut_LG_2)*dx**2)

#I_sat_LG = 1/saturation_factor*np.max(ideal_donut_LG)
#I_sat_LG_2 = 1/saturation_factor*np.max(ideal_donut_LG_2)
I_sat = 1/saturation_factor*np.max(ideal_donut_LG)

PSF_vs_depth_LG = np.zeros((len(LG[0].depths),num_runs))
#PSF_vs_depth_LG = np.zeros((len(LG[0].depths),num_runs))
PSF_centroid_LG = np.zeros(PSF_vs_depth_LG.shape)

PSF_vs_depth_LG_2 = np.zeros(PSF_vs_depth_LG.shape)
PSF_centroid_LG_2 = np.zeros(PSF_vs_depth_LG.shape)

STED_power_LG = np.zeros(PSF_vs_depth_LG.shape)
STED_power_LG_2 = np.zeros(PSF_vs_depth_LG.shape)

for run_number in range(num_runs):
    for depth_index in range(len(depths)):
        # Note: The absorption mean free path is vastly bigger than scattering mean free path
        # The scattering angles are also small in tissue, so photons are unlikely to be scattered out of the simulation volume.
        # Therefore, I'm assuming that the total optical power in the finite difference simulation volume is conserved for every transverse cross section. This should be true to within a small margin of error
        # However, I find that the total power calculated at the focual plane is sometimes a little larger than what was sent in. This is most likely due to errors from discretization.
        # So I'm normalizing power at focus before calculating the STED FWHM

        attenuation_factor = np.exp(-depths[depth_index]/ls)


        field_profile_LG = LG[run_number].intensity_profiles[depth_index]
        #field_profile_LG = field_profile_LG/(np.sum(field_profile_LG)*dx**2)
        PSF_vs_depth_LG[depth_index,run_number], PSF_centroid_LG[depth_index,run_number], STED_power_LG[depth_index,run_number] = STED_psf_fwhm(dx,attenuation_factor*excitationBeam,field_profile_LG, I_sat)
        
        field_profile_LG_2 = LG_2[run_number].intensity_profiles[depth_index]
        #field_profile_LG_2 = field_profile_LG_2/(np.sum(field_profile_LG_2)*dx**2)
        PSF_vs_depth_LG_2[depth_index,run_number], PSF_centroid_LG_2[depth_index,run_number], STED_power_LG_2[depth_index,run_number] = STED_psf_fwhm(dx,attenuation_factor*excitationBeam,field_profile_LG_2, I_sat/LG2_LG1_power_ratio)


        '''
        plt.pcolormesh(excitationBeam>fluorescenceThreshold)
        plt.show()
        plt.pcolormesh(field_profile_LG<I_sat)
        plt.show()
        plt.pcolormesh(np.logical_and((excitationBeam>fluorescenceThreshold),(field_profile_LG<I_sat)))
        plt.show()
        '''
10

LG_figure_of_merit = np.mean(PSF_vs_depth_LG,axis=1)
LG_2_figure_of_merit = np.mean(PSF_vs_depth_LG_2,axis=1)
LG_deviation = np.median(PSF_centroid_LG,axis=1)
LG_2_deviation = np.median(PSF_centroid_LG_2,axis=1)
power_LG = np.mean(STED_power_LG,axis=1)
power_LG_2 = np.mean(STED_power_LG_2,axis=1)

# Ideal case at surface
# Note: Does not use Debye-Wolf integral. There might be a very small error.

'''
ideal_PSF_diameter, _, ideal_STED_power = STED_psf_fwhm(dx,excitationBeam,2*ideal_donut_LG, I_sat)
depths[0] = 0
LG_figure_of_merit[0] = ideal_PSF_diameter
LG_2_figure_of_merit[0] = ideal_PSF_diameter
LG_deviation[0] = 0
LG_2_deviation[0] = 0
power_LG[0] = ideal_STED_power
power_LG_2[0] = ideal_STED_power
depths = np.concatenate(([0],depths))
LG_figure_of_merit = np.concatenate(([ideal_PSF_diameter],LG_figure_of_merit))
LG_2_figure_of_merit = np.concatenate(([ideal_PSF_diameter],LG_2_figure_of_merit))
LG_deviation = np.concatenate(([0],LG_deviation))
LG_2_deviation = np.concatenate(([0],LG_2_deviation))
power_LG = np.concatenate(([ideal_STED_power],power_LG))
power_LG_2 = np.concatenate(([ideal_STED_power],power_LG_2))
'''

plt.rcParams.update({'font.size': 20})
plt.rcParams['pcolor.shading'] = 'auto'
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 2

plt.figure(figsize=[16,4],dpi=500)
plt.subplot(1,3,1)
plt.plot(depths,LG_figure_of_merit,label='LG', marker = 'o')
plt.plot(depths,LG_2_figure_of_merit,label='LG_2', marker = 'o')
plt.plot([2,25,50,75],[113,137,177,204],label='LG, Expt.',linestyle='None',marker = 'D')
plt.legend(loc='lower left')
plt.xlabel('Tissue depth (μm)')
plt.ylabel('FWHM (nm)')
plt.ylim(bottom = 0)

plt.subplot(1,3,2)
plt.plot(depths,LG_deviation,label='LG', marker = 'o')
plt.plot(depths,LG_2_deviation,label='LG_2', marker = 'o')
plt.legend(loc='lower left')
plt.ylim(bottom = 0)
plt.xlabel('Tissue depth (μm)')
plt.ylabel('PSF deviation (nm)')

plt.subplot(1,3,3)
plt.plot(depths,100/8*np.exp(-depths/ls)*power_LG,label='LG', marker = 'o')
plt.plot(depths,100/8*np.exp(-depths/ls)*power_LG_2,label='LG_2', marker = 'o')
plt.legend(loc='lower left')
plt.ylim(bottom = 0)
plt.xlabel('Tissue depth (μm)')
plt.ylabel('STED Power (a.u.)')
plt.tight_layout()
plt.show()

#####
plt.rcParams.update({'font.size': 18})
plt.rcParams['pcolor.shading'] = 'auto'
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 2

plt.figure(figsize=[12,4],dpi=500)
plt.subplot(1,2,1)
plt.plot(depths[0:7],LG_figure_of_merit[0:7],label='LG, L=1', marker = 'o')
plt.plot(depths[0:7],LG_2_figure_of_merit[0:7],label='LG, L=2', marker = 'D')
#plt.plot([2,25,50,75],[113,137,177,204],label='LG, Expt. [10]',linestyle='None',marker = 's')
plt.legend(loc='lower right')
plt.xlabel('Tissue depth (μm)')
plt.ylabel('FWHM (nm)')
plt.ylim([0,300])

plt.subplot(1,2,2)
plt.plot(depths[0:7],30+10*np.log10(np.exp(-depths[0:7]/ls)*power_LG[0:7]),label='LG, Sim.', marker = 'o')
plt.plot(depths[0:7],30+10*np.log10(np.exp(-depths[0:7]/ls)*power_LG_2[0:7]),label='LG_2, Sim.', marker = 'D')
plt.legend(loc='lower left')
#plt.ylim(bottom = 0)
plt.xlabel('Tissue depth (μm)')
plt.ylabel('SNR (dB)')
plt.tight_layout()
plt.show()