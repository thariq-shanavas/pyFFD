import numpy as np
from FriendlyFourierTransform import FFT2
from scipy.interpolate import RegularGridInterpolator
import warnings

def TightFocus(InputField_x,InputField_y,dx,wavelength,n_homogenous,FocusDepth,MeasurementPlane_z=0,target_dx=25e-9):
    '''
    Provides the 3D field from focusing a polarized input field through a thick lens
    Axis of the lens is along z. Input polarization is assumed to be along x direction.
    Method described in: [1] https://doi.org/10.1016/j.optcom.2019.06.022 and [2] https://doi.org/10.1364/OE.14.011277 
    Important: The Debye approximation is best in a z-plane near the focus. It's worse the farther you are from the focus.
    Tip: If the output pixel resolution is too low, try increasing dx or xy_cells

    Inputs:
    Input Field: n-by-n matrix containing the spatial distribution of Ex
    dx: Discretization of space
    FocusDepth: Focal length of the lens measured in the medium, i.e., depth at which light is focused
    Measurement Plane z: z-coordinate of the xy-plane at which the field output is desired, measured from focus.
    n_homogenous: The refractive index of the medium, which is assumed to be homogenous
    wavelength: wavelength of the light

    Outputs:
    
    Ex, Ey, Ez: Electric field components at z = MeasurementPlane_z
    '''

    '''
    Methodology:

    Step 1: Initialization
    Calculate n-by-n matrices theta and phi using FocusDepth, xy_cells and dx
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
    '''
    MeasurementPlane_z = -MeasurementPlane_z    # Origin is at focus and z axis is along the direction of propagation. See Fig 1, [1]
    xy_cells = np.shape(InputField_x)[0]
    indices = np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
    xx, yy = np.meshgrid(dx*indices,dx*indices)
    k = 2*np.pi/wavelength*n_homogenous
    R = (xy_cells/2)*dx
    if R>FocusDepth:
        wrn = 'The implementation of the Debye-Wolf integral has shown some scaling issues in the very high NA regime. Exercise caution!'
        warnings.warn(wrn)
    NA = n_homogenous*(R/np.sqrt(R**2+FocusDepth**2))     # Obviously not the actual NA, but this is used for mapping k-space to real space later
    
    # Check if debye-Wolf method is valid (Eq. 13.13 [https://doi.org/10.1016/0030-4018(81)90107-3])
    if not(k*FocusDepth>10*np.pi/(np.sin(0.5*np.arcsin(NA/n_homogenous)))**2):
        raise ValueError('Debye-Wolf Integral is not valid. See [https://doi.org/10.1016/0030-4018(81)90107-3]')

    # Angles in the input plane, measured from the focus
    #theta = np.arctan(np.sqrt(xx**2+yy**2)/FocusDepth)
    
    theta = np.arcsin(np.minimum(0.9999,(np.sqrt(xx**2+yy**2)/R)*NA/n_homogenous))  # theta has to be less than pi/2: There is division by zero in the FFT integrand otherwise.
    phi = np.arctan2(yy,xx)

    # Input fields in cylindrical coordinates
    # Assumption: Input is polarized along positive x direction.
    # https://en.wikipedia.org/wiki/Vector_fields_in_cylindrical_and_spherical_coordinates
    # phi is the same as calculated earlier in the input plane
    A_0rho = InputField_x*np.cos(phi) + InputField_y*np.sin(phi)
    A_0phi = -InputField_x*np.sin(phi) + InputField_y*np.cos(phi)

    # Angular spectrum of input field in cartesian coordinates
    # Note: There is a disagreement in sign between [1] and [2] for Az. Following [2]
    # The disagreement in sign is because [1] uses FFT without a negative sign for kx and ky
    # These are the fields *evaluated on a curved surface* centered at the focus and radius of curvature = focal length
    # They are sampled non-uniformly in the curved surface, but uniformly in the projection of the curved surface on the xy plane
    # See fig. 1 in Ref [2]

    Ax = np.sqrt(np.cos(theta))*(np.cos(theta)*np.cos(phi)*A_0rho - np.sin(phi)*A_0phi)
    Ay = np.sqrt(np.cos(theta))*(np.cos(theta)*np.sin(phi)*A_0rho + np.cos(phi)*A_0phi)
    Az = np.sqrt(np.cos(theta))*(np.sin(theta)*A_0rho)

    # Eq. 6, Ref. [2]
    kx = -k*np.cos(phi)*np.sin(theta)
    ky = -k*np.sin(phi)*np.sin(theta)
    kz = k*np.cos(theta)

    # Debye-Wolf Integral
    # Important note: As per [2], we integrate in the d(theta).d(phi) domain. The E-field is not uniformly sampled in theta and phi.
    # But [1] transforms the variable of integration to a uniform grid in the kx-ky space.
    # This makes it easy to evaluate it as an FFT. See eq. 4, Ref [1]
    # The domain is not the whole k-space. However, the integrand vanishes outside the domain of interest, so we can still use FFT

    # Ax, Ay, Az were originally evaluated in a curved plane with non-uniform sampling in theta and phi.
    # Equivalently, sampled uniformly in a grid spaced by dx in a plane projection.
    # If we can cransform the discretization dx to dk, Ax becomes uniformly sampled in kx-ky space. We need this to run FFT.
    # The plane projection with N samples in the x-direction is mapped to 2.k.sin(theta_max) in the x-direction. i.e., -k.sin(theta) to k.sin(theta)
    # sin(theta_max) is related to the NA of the lens as NA/index
    # Therefore kx is discretized as (d kx) = (2.k.{NA/n_homogenous})/xy_cells

    d_kx = 2*k*NA/(n_homogenous*xy_cells)       # This is probably wrong by a factor of 2 pi. See below.
    
    # After the FFT, the output space is descretized as 1/(d_kx*xy_cells) = wavelength/(4*pi*NA)
    # Note that this is not really the NA of the lens, rather, related to the max. angle subtended by the input plane at focus.
    # Actual simulation suggests that the output dx is actually wavelength/(2*NA) I'm not sure how I'm missing a factor of 2.pi
    
    out_dx = wavelength/(2*NA) 
    #xy_span = xy_cells*dx/2
    #out_dx = wavelength/(2*n_homogenous*(xy_span/np.sqrt(xy_span**2+FocusDepth**2)))    # lambda/(2.n.sin(theta_max))
    prefactor = -1j*4*R**2*k/(wavelength*FocusDepth*xy_cells**2)

    Ex = prefactor*FFT2(np.exp(1j*kz*MeasurementPlane_z)*Ax/kz)
    Ey = prefactor*FFT2(np.exp(1j*kz*MeasurementPlane_z)*Ay/kz)
    Ez = prefactor*FFT2(np.exp(1j*kz*MeasurementPlane_z)*Az/kz)

    # Normalization
    n0 = np.sum(np.abs(Ex)**2+np.abs(Ey)**2+np.abs(Ez)**2)*out_dx**2
    Ex = Ex/n0
    Ey = Ey/n0
    Ez = Ez/n0

    # Scaling the discretization in space
    ScalingFactor = target_dx/out_dx
    if ScalingFactor <0.1:
        wrn = 'High output interpolation factor ('+str(1/ScalingFactor)+'): Increase xy_cells or dx'
        warnings.warn(wrn)

    interpEx = RegularGridInterpolator((out_dx*indices,out_dx*indices), Ex, bounds_error = False, fill_value = 0)
    interpEy = RegularGridInterpolator((out_dx*indices,out_dx*indices), Ey, bounds_error = False, fill_value = 0)
    interpEz = RegularGridInterpolator((out_dx*indices,out_dx*indices), Ez, bounds_error = False, fill_value = 0)
    dx_new = out_dx*ScalingFactor
    xx_new, yy_new = np.meshgrid(dx_new*indices,dx_new*indices,indexing='ij')

    # TODO: Ex, Ey, Ez needs to be scaled by scalingFactor^2 for the total power integrated in new coordinate system to be 1
    return interpEx((xx_new,yy_new)),interpEy((xx_new,yy_new)),interpEz((xx_new,yy_new)),dx_new
    #return Ex,Ey,Ez, out_dx
