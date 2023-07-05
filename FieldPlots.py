# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

'''
import seaborn as sns
def PlotSnapshots(Field_snapshots, imaging_depth):
    f,((ax0,ax1),(ax2,ax3)) = plt.subplots(2,2)
    g0 = sns.heatmap(np.abs(Field_snapshots[:,:,0]), ax=ax0)
    g0.set_title('Seed')
    g1 = sns.heatmap(np.abs(Field_snapshots[:,:,1]), ax=ax1)
    g1.set_title('%1.1f $\mu m$' %(10**6*imaging_depth[0]))
    g2 = sns.heatmap(np.abs(Field_snapshots[:,:,2]), ax=ax2)
    g2.set_title('%1.1f $\mu m$' %(10**6*imaging_depth[1]))
    g3 = sns.heatmap(np.abs(Field_snapshots[:,:,3]), ax=ax3)
    g3.set_title('%1.1f $\mu m$' %(10**6*imaging_depth[2]))
    return f
'''

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
        