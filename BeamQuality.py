import numpy as np

def STED_psf_area(dx,excitationBeam,depletionBeam, fluorescenceThreshold, I_sat):
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
    fluorophore_active = ((excitationBeam>fluorescenceThreshold) and (depletionBeam<I_sat))
    num_active_fluorphores = fluorophore_active.count(True)

    # pi*r**2 = num_active_fluorphores*dx**2
    # Return diameter = 2*r
    return 2*np.sqrt(num_active_fluorphores*dx**2/np.pi)
    