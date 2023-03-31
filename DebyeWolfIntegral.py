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
    Measurement Plane z: z-coordinate of the xy-plane at which the field output is desired, measured from focus.
    n_homogenous: The refractive index of the medium, which is assumed to be homogenous
    wavelength: wavelength of the light

    Outputs:
    
    Ex, Ey, Ez: Electric field components at z = MeasurementPlane_z
    '''

    '''
    Methodology:

    Step 1: Initialization
    Calculate n-by-n matrices theta and phi using FocalLength, xy_cells and dx
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
    The integral is the form of an FFT which makes computations easier. 
    i.e., for constant z, Eq. 4 RHS is the FFT of exp(i.kz.z) A(θ,φ) / kz
    '''

    xy_cells = np.shape(InputField)[0]
    indices = np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
    xx, yy = np.meshgrid(dx*indices,dx*indices)

    # Angles in the input plane, measured from the focus
    theta = np.arcsin(np.sqrt(xx**2+yy**2)/FocalLength)
    phi = np.arctan2(yy,xx)

    # Input fields in cylindrical coordinates
    # Assumption: Input is polarized along positive x direction.
    # https://en.wikipedia.org/wiki/Vector_fields_in_cylindrical_and_spherical_coordinates
    # phi is the same as calculated earlier in the input plane
    A_0rho = InputField*np.cos(phi)
    A_0phi = -InputField*np.sin(phi)

    # Angular spectrum of input field in cartesian coordinates
    Ax = np.sqrt(np.cos(theta))*(np.cos(theta)*np.cos(phi)*A_0rho - np.sin(phi)*A_0phi)
    Ay = np.sqrt(np.cos(theta))*(np.cos(theta)*np.sin(phi)*A_0rho + np.cos(phi)*A_0phi)
    Az = -np.sqrt(np.cos(theta))*np.sin(theta)*A_0phi

    # Debye-Wolf Integral
    fx = fy = 1/(dx*xy_cells)*np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
    Ex = 