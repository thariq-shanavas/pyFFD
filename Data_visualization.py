import numpy as np
import matplotlib.pyplot as plt
from ParallelTightFocusContrastVsDepth import Results, Tightfocus_HG, Tightfocus_LG
from BeamQuality import STED_psf_fwhm
from DebyeWolfIntegral import SpotSizeCalculator
from SeedBeams import Gaussian_beam

# VERY IMPORTANT NOTE
# Do not rely on the order of items in contrasts or intensity_profile.
# Match the depth list object to any result list item.

# Parameters
# Saturation factor is the peak power of an ideal donut over the saturation power, for either type.
# To simulate increasing donut power, increase this factor.

saturation_factor = 20

LG = np.load('Results/Contrast_LG.npy', allow_pickle=True)
#HG = np.load('Results/Contrast_HG.npy', allow_pickle=True)
plot_contrast = False
plot_STED_PSF_fwhm = True
fast_mode = False
num_runs = len(LG)

if plot_contrast:
    contrast_vs_depth_LG = np.zeros((len(LG[0].depths),num_runs))
    #contrast_vs_depth_HG = np.zeros((len(HG[0].depths),num_runs))
    for i in range(num_runs):
        contrast_vs_depth_LG[:,i] = LG[i].contrasts
        #contrast_vs_depth_HG[:,i] = HG[i].contrasts

    plt.plot(LG[0].depths,np.median(contrast_vs_depth_LG,axis=1),label='LG')
    #plt.plot(HG[0].depths,np.median(contrast_vs_depth_HG,axis=1),label='HG')
    plt.legend()


if plot_STED_PSF_fwhm:
    dx = LG[0].dx
    xy_cells = LG[0].intensity_profile[0].shape[0]
    depths = LG[0].depths * 10**6
    excitation_spot_size = SpotSizeCalculator(LG[0].focus_depth,LG[0].beam_radius,LG[0].n_h,LG[0].wavelength,0) # 1/e^2 diameter of a gaussian at focus
    excitationBeam = (Gaussian_beam(xy_cells,dx,excitation_spot_size/2))**2
    fluorescenceThreshold = (1/2.71**2)*np.max(excitationBeam)
    _,_, ideal_donut_LG = Tightfocus_LG([0,'',0])
    #_,_, ideal_donut_HG = Tightfocus_HG([0,'',0])

    # Normalizing power to unity
    ideal_donut_LG = ideal_donut_LG/(np.sum(ideal_donut_LG)*dx**2)
    #ideal_donut_HG = ideal_donut_HG/(np.sum(ideal_donut_HG)*dx**2)

    #I_sat_LG = 1/saturation_factor*np.max(ideal_donut_LG)
    #I_sat_HG = 1/saturation_factor*np.max(ideal_donut_HG)
    I_sat = 1/saturation_factor*np.max(ideal_donut_LG)

    PSF_vs_depth_LG = np.zeros((len(LG[0].depths),num_runs))
    #PSF_vs_depth_HG = np.zeros(PSF_vs_depth_LG.shape)
    for run_number in range(num_runs):
        for depth_index in range(len(depths)):
            # Note: The absorption mean free path is vastly bigger than scattering mean free path
            # The scattering angles are also small in tissue, so photons are unlikely to be scattered out of the simulation volume.
            # Therefore, I'm assuming that the total optical power in the finite difference simulation volume is conserved for every transverse cross section. This should be true to within a small margin of error
            # However, I find that the total power calculated at the focual plane is sometimes a little larger than what was sent in. This is most likely due to errors from discretization.
            # So I'm normalizing power at focus before calculating the STED FWHM

            field_profile_LG = LG[run_number].intensity_profile[depth_index]
            #field_profile_LG = field_profile_LG/(np.sum(field_profile_LG)*dx**2)
            PSF_vs_depth_LG[depth_index,run_number] = 10**9*STED_psf_fwhm(dx,excitationBeam,field_profile_LG,fluorescenceThreshold , I_sat, fast_mode)

            #field_profile_HG = HG[run_number].intensity_profile[depth_index]
            #field_profile_HG = field_profile_HG/(np.sum(field_profile_HG)*dx**2)
            #PSF_vs_depth_HG[depth_index,run_number] = 10**9*STED_psf_fwhm(dx,excitationBeam,field_profile_HG,fluorescenceThreshold , I_sat, fast_mode)
            '''
            plt.pcolormesh(excitationBeam>fluorescenceThreshold)
            plt.show()
            plt.pcolormesh(field_profile_LG<I_sat)
            plt.show()
            plt.pcolormesh(np.logical_and((excitationBeam>fluorescenceThreshold),(field_profile_LG<I_sat)))
            plt.show()
            '''
    LG_figure_of_merit = np.median(PSF_vs_depth_LG,axis=1)
    #HG_figure_of_merit = np.median(PSF_vs_depth_HG,axis=1)
    fig,ax = plt.subplots()
    plt.plot(depths,LG_figure_of_merit,label='LG', marker = 'o')
    detection_count_LG = np.count_nonzero(PSF_vs_depth_LG,axis=1)
    '''
    for i, txt in enumerate(detection_count_LG):
        ax.annotate(txt, (depths[i], LG_figure_of_merit[i]-5))
    
    plt.plot(depths,HG_figure_of_merit,label='HG', marker = 'o')

    detection_count_HG = np.count_nonzero(PSF_vs_depth_HG,axis=1)
    for i, txt in enumerate(detection_count_HG):
        ax.annotate(txt, (depths[i], HG_figure_of_merit[i]-5))
    '''
    plt.legend()

    plt.ylim(bottom = 0)
    plt.gcf().set_dpi(300)
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['pcolor.shading'] = 'auto'
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.linewidth"] = 2
    plt.title("STED PSF for LG vs HG depletion beam", weight='bold')
    plt.xlabel("Tissue depth ($µm$)", weight='bold', fontsize=12)
    plt.xticks(weight = 'bold', fontsize=12)
    plt.ylabel("STED PSF fwhm ($nm$)", weight='bold', fontsize=12)
    plt.yticks(weight = 'bold', fontsize=12)
    plt.tight_layout()
    plt.show()