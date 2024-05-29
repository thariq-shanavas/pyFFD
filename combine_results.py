import numpy as np
import copy
from helper_classes import Parameters_class,Results_class

LG1 = np.load('Results.750nm.5-50/Contrast_LG.npy', allow_pickle=True)
LG2 = np.load('Results.750nm.55-80/Contrast_LG.npy', allow_pickle=True)
HG1 = np.load('Results.750nm.5-50/Contrast_LG.npy', allow_pickle=True)
HG2 = np.load('Results.750nm.55-80/Contrast_LG.npy', allow_pickle=True)

num_runs = len(LG1)
depths = np.array([5e-6,10e-6,15e-6,20e-6,25e-6,30e-6,35e-6,40e-6,45e-6,50e-6,55e-6,60e-6,65e-6,70e-6,75e-6,80e-6])
depths1 = np.array([5e-6,10e-6,15e-6,20e-6,25e-6,30e-6,35e-6,40e-6,45e-6,50e-6])
depths2 = np.array([55e-6,60e-6,65e-6,70e-6,75e-6,80e-6])

LG_result = []              # Results will be stored in a list, with each item being an object of class 'Results'
HG_result = []
parameters = Parameters_class(depths,LG1[0].dx_for_export,LG1[0].wavelength,LG1[0].max_FDFD_dx,LG1[0].resolution_factor,LG1[0].FDFD_dz,LG1[0].beam_radius,LG1[0].focus_depth,LG1[0].unique_layers,LG1[0].n_h,LG1[0].ls,LG1[0].g)

for run_number in range(num_runs):
    tmp_field_exports_LG  = []
    tmp_field_exports_HG  = []
    tmp_index1 = 0
    tmp_index2 = 0
    
    for depth in depths:
        
        if depth in depths1:
            tmp_field_exports_LG.append(LG1[run_number].intensity_profiles[tmp_index1])    
            tmp_field_exports_HG.append(HG1[run_number].intensity_profiles[tmp_index1])    
            tmp_index1 = tmp_index1+1

        if depth in depths2:
            tmp_field_exports_LG.append(LG2[run_number].intensity_profiles[tmp_index2])    
            tmp_field_exports_HG.append(HG2[run_number].intensity_profiles[tmp_index2])    
            tmp_index2 = tmp_index2+1
    

    # Save results. The Results object is mutable in Python, it needs to be deep-copied
    LG_result.append(copy.deepcopy(Results_class(parameters,tmp_field_exports_LG)))
    HG_result.append(copy.deepcopy(Results_class(parameters,tmp_field_exports_HG)))

    np.save('Results/Contrast_LG', LG_result)
    np.save('Results/Contrast_HG', HG_result)