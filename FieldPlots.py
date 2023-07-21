# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from DebyeWolfIntegral import SpotSizeCalculator

def plot_HG(focus_depth,beam_radius,n_h,wavelength,xy_cells,FDFD_dx,HG10_Focus_Intensity,HG01_Focus_Intensity,Focus_Intensity,FDFD_depth,run_number):
    fig = plt.figure()
    plt.gcf().set_dpi(500)
    plt.rcParams.update({'font.size': 5})
    plt.rcParams['pcolor.shading'] = 'auto'
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.linewidth"] = 1
    
    ax1 = fig.add_subplot(1,3,1, adjustable='box', aspect=1)
    ax2 = fig.add_subplot(1,3,2, adjustable='box', aspect=1)
    ax3 = fig.add_subplot(1,3,3, adjustable='box', aspect=1)

    # Parameters for saving the images to Results folder.
    spot_size_at_focus = SpotSizeCalculator(focus_depth,beam_radius,n_h,wavelength,0)   # For plotting
    imaging_dx = spot_size_at_focus*6/xy_cells
    indices = np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
    axis = 10**6*FDFD_dx*indices
    imaging_axes = 10**6*imaging_dx*indices
    xx_imaging, yy_imaging = np.meshgrid(imaging_axes,imaging_axes, indexing='ij')  # Mesh grid used for plotting

    ax1.pcolormesh(imaging_axes,imaging_axes,RegularGridInterpolator((axis,axis),HG10_Focus_Intensity, bounds_error = True, method='linear')((xx_imaging, yy_imaging)))
    ax1.set_title('HG 10 beam at focus', fontweight='bold')
    ax1.set_xlabel("x ($µm$)", fontweight='bold')
    ax1.set_ylabel("y ($µm$)", fontweight='bold')

    ax2.pcolormesh(imaging_axes,imaging_axes,RegularGridInterpolator((axis,axis),HG01_Focus_Intensity, bounds_error = True, method='linear')((xx_imaging, yy_imaging)))
    ax2.set_title('HG 01 beam at focus', fontweight='bold')
    ax2.set_xlabel("x ($µm$)", fontweight='bold')
    ax2.set_ylabel("y ($µm$)", fontweight='bold')

    ax3.pcolormesh(imaging_axes,imaging_axes,RegularGridInterpolator((axis,axis),Focus_Intensity , bounds_error = True, method='linear')((xx_imaging, yy_imaging)))
    ax3.set_title('HG 01 + HG 10 (Incoherent Donut)', fontweight='bold')
    ax3.set_xlabel("x ($µm$)", fontweight='bold')
    ax3.set_ylabel("y ($µm$)", fontweight='bold')

    plt.tight_layout()
    plt.savefig('Results/HG_'+str("{:02d}".format(int(1e6*FDFD_depth)))+'um_run'+str("{:02d}".format(run_number))+'.png', bbox_inches = 'tight', dpi=500)
    plt.close()

def plot_LG(focus_depth,beam_radius,n_h,wavelength,xy_cells,FDFD_dx,LG_Focus_Intensity,FDFD_depth,run_number):
    
    fig = plt.figure()
    plt.gca().set_aspect('equal')
    plt.gcf().set_dpi(500)
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['pcolor.shading'] = 'auto'
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.linewidth"] = 2

    spot_size_at_focus = SpotSizeCalculator(focus_depth,beam_radius,n_h,wavelength,0)   # For plotting
    imaging_dx = spot_size_at_focus*6/xy_cells
    indices = np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)
    axis = 10**6*FDFD_dx*indices
    imaging_axes = 10**6*imaging_dx*indices
    xx_imaging, yy_imaging = np.meshgrid(imaging_axes,imaging_axes, indexing='ij')  # Mesh grid used for plotting

    plt.pcolormesh(imaging_axes,imaging_axes,RegularGridInterpolator((axis,axis),LG_Focus_Intensity, bounds_error = True, method='linear')((xx_imaging, yy_imaging)))
    plt.title("LG beam at focus", weight='bold')
    plt.xlabel("x ($µm$)", weight='bold', fontsize=12)
    plt.xticks(weight = 'bold', fontsize=12)
    plt.ylabel("y ($µm$)", weight='bold', fontsize=12)
    plt.yticks(weight = 'bold', fontsize=12)

    plt.tight_layout()
    plt.savefig('Results/LG_'+str("{:02d}".format(int(1e6*FDFD_depth)))+'um_run'+str("{:02d}".format(run_number))+'.png', bbox_inches = 'tight', dpi=500)
    plt.close()



def VortexNull(Field, dx, beam_type, cross_sections = 19, num_samples = 1000, filter_sigma = 1):
    
    # Important: This function takes field intensity, not E-field
    # Takes n radial cross sections at equal angles to find the average radial profile. n = cross_sections.
    # Takes num_samples samples at each radial cross section.
    
    # Find argument of max value in the field
    xy_cells = np.shape(Field)[0]
    Field = np.abs(Field.astype(float))

    if beam_type=='LG' or beam_type=='HG':
        posx,posy = np.unravel_index(Field.argmax(), Field.shape)
        posx = posx - xy_cells/2
        posy = posy - xy_cells/2
        beam_radius = dx*np.sqrt(posx**2+posy**2) # distance of peak field from origin
    elif beam_type=='G':
        # Quick and dirty implementation
        midpoint = int(xy_cells/2)
        beam_cross_section = Field[midpoint,:]
        tmp_index = int(xy_cells/2)
        while beam_cross_section[tmp_index]>beam_cross_section[midpoint]/2.718**2:
            tmp_index = tmp_index+1
        beam_radius = dx*(tmp_index-midpoint-0.5)
    else:
        ValueError('Beam type unknown.')

    indices = np.linspace(-xy_cells/2,xy_cells/2-1,xy_cells,dtype=np.int_)

    # Defining an interpolation function to be used later
    # Interpolation is defined using a coordinate system with origin at the middle.
    Field_interpolation = RegularGridInterpolator((dx*indices, dx*indices), Field, method='linear', bounds_error = True)

    if beam_type=='LG' or beam_type=='HG':
        # Find the coordinates of the center of the null, within a central 3x3 square
        central_3x3 = Field[int(xy_cells/2-1):int(xy_cells/2+2),int(xy_cells/2-1):int(xy_cells/2+2)]
        arg_min = np.unravel_index(central_3x3.argmin(), central_3x3.shape)
        beam_center_x = dx*(arg_min[0] - 1)
        beam_center_y = dx*(arg_min[1] - 1)
        if Field[int(xy_cells/2 + arg_min[0]-1),int(xy_cells/2 + arg_min[0]-1)] < 1e-30:
            ValueError(" E field is zero at null, hence there is unlimited contrast. This should not have happened.")
        
    elif beam_type=='G':
        smoothened_field = gaussian_filter(Field.astype(float), sigma=filter_sigma)
        beam_center_y,beam_center_x = np.unravel_index(np.abs(smoothened_field).argmax(), np.abs(smoothened_field).shape)
        beam_center_x = dx*(beam_center_x - xy_cells/2)
        beam_center_y = dx*(beam_center_y - xy_cells/2)

    contrasts = np.zeros(cross_sections)

    for i in range(cross_sections):
        
        theta = (2*np.pi/cross_sections)*i
        xp = np.linspace(0,2*beam_radius,num_samples)*(np.cos(theta))+beam_center_x
        yp = np.linspace(0,2*beam_radius,num_samples)*(np.sin(theta))+beam_center_y
        Field_cross_section = Field_interpolation((xp,yp))
        contrasts[i] = 10*np.log10(np.max(Field_cross_section)/Field_cross_section[0])
    
    null_contrast = np.mean(contrasts)
    # print('Null contrast is %1.1f' %(null_contrast))
    # print('Beam center at %1.1f nm, %1.1f nm' %(beam_center_x*10**9,beam_center_y*10**9))
    # print('Spot size is %1.1f nm' %(2*beam_radius*10**9))
    return null_contrast, np.std(contrasts)
        