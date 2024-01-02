import numpy as np
from getFWHM_2D import getFWHM_GaussianFitScaledAmp

def STED_psf_fwhm(dx,excitationBeam,depletionBeam,I_sat):
    # This function approximately calculates the spot size of the STED point spread function

    ## If the excitationBeam is brighter than fluorescenceThreshold, the fluorophore is excited.
    ## If the depletionBeam is brighter than I_sat, the fluoroscence is suppressed.

    # For example, we may assume that the excitation beam is gaussian, and all fluorophores within 1/e^2 intensity of the excitation beam maximum is excited.
    # We assume that in the ideal case, the STED PSF has a spot size of 50 nm.
    # In that case, set 
    # fluorescenceThreshold = 1/e^2 * max(excitationBeam)
    # I_sat = depletionBeam[xy_cells/2, xy_cells/2+ int(25e-9/dx)]

    
    # Make sure dx is small enough
    if dx > 10e-9:
        ValueError("Interpolate the function before calculating STED PSF!")

    # More realistic estimate of FWHM
    # Following https://doi.org/10.1364/OE.16.004154
    # depletionBeam = np.abs(depletionBeam)/(np.sum(np.abs(depletionBeam))*dx**2)         # This step is a workaround for the power instability. See github issue.
    # excitationBeam = np.abs(excitationBeam)/np.sum(np.abs(excitationBeam)*dx**2)
    eta = np.exp(-np.log(2)*np.abs(depletionBeam)/I_sat)    
    STED_psf = excitationBeam*eta
    STED_power = np.sum(np.abs(STED_psf)*dx**2)

    ## If you use the same formula on a gaussian, you get the FWHM. This is a heuristic.
    # pi*(FWHM/2)^2 = dx^2 * np.sum(STED_psf>np.max(STED_psf)/2)
    PSF_diameter = 2*dx*np.sqrt(np.sum(STED_psf>np.max(STED_psf)/2)/3.14)

    xy_cells = np.shape(depletionBeam)[0]
    axes = dx*np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
    xx, yy = np.meshgrid(axes,axes, indexing='ij')
    centroid_x = np.average(xx,weights = STED_psf)
    centroid_y = np.average(yy,weights = STED_psf)
    centroid_deviation = np.sqrt(centroid_x**2+centroid_y**2)
    ## Fit gaussian     
    # PSF_diameter = dx*getFWHM_GaussianFitScaledAmp(STED_psf)
    return 10**9*PSF_diameter, 10**9*centroid_deviation, STED_power