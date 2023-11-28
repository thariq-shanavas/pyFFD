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