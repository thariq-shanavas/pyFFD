import numpy as np


def TightFocus(InputField,dx,wavelength,n_homogenous,FocalLength,NA,MeasurementPlane_z):
    '''
    Provides the 3D field from focusing a polarized input field through a thick lens
    Axis of the lens is along z. Input polarization is assumed to be along x direction.
    Method described in: https://doi.org/10.1016/j.optcom.2019.06.022

    Inputs:
    Input Field: n-by-n matrix containing the spatial distribution of Ex
    dx: Discretization of space
    Focal Length: Focal length of the lens
    NA: Numerical aperture. While this can be calculated from Input Field and focal length, providing it as an argument simplifies calculation.
    Measurement Plane z: z-coordinate of the xy-plane at which the field output is desired
    n_homogenous: The refractive index of the medium, which is assumed to be homogenous
    wavelength: wavelength of the light

    Outputs:
    
    Ex, Ey, Ez: Electric field components at z = MeasurementPlane_z
    '''

    '''
    Methodology:

    Step 1: Initialization
    Calculate n-by-n matrices theta and phi using FocalLength and MeasurementPlane_z
    Calculate n-by-n matrices A_0rho and A_0phi from InputField
    Calculate n-by-n matrices Ax, Ay, Az from A_0rho, A_0phi, theta and phi using Eq. 5
    
    Step 2: Debye-Wolf integral

    Integral is performed in the domain (kx,ky): kx^2 + ky^2 < [k sin(theta_max)]^2.
    RHS = (2.pi.NA / wavelength)^2. To respect this domain of integration, we multiply by a mask of ones and zeros and integrate in the whole kx-ky space.

    kx and ky are n-by-n grids, with 
    kx[0,0] = -k.sin(θ)
    kx[n,0] = -k.sin(θ)
    kx[0,n] = +k.sin(θ)
    kx[n,n] = +k.sin(θ)

    ky[0,0] = -k.sin(θ)
    ky[n,0] = +k.sin(θ)
    ky[0,n] = -k.sin(θ)
    ky[n,n] = +k.sin(θ)

    kz = sqrt(k^2 - kx^2 - ky^2)

    For each x,y, calculate three integrals for Ex, Ey, Ez
    Unfortunately, we need to do this in a for loop for all x,y values: about 1 million (x,y) pairs, for each of the three polarizations.
    '''