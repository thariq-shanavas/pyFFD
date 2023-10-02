class Parameters_class:
    def __init__(self,depths,dx_for_export,wavelength,max_FDFD_dx,resolution_factor,FDFD_dz,beam_radius,focus_depth,unique_layers,n_h,ls,g):
        self.depths = depths
        self.dx_for_export = dx_for_export
        self.wavelength = wavelength
        self.max_FDFD_dx = max_FDFD_dx
        self.resolution_factor = resolution_factor
        self.FDFD_dz = FDFD_dz
        self.beam_radius = beam_radius
        self.focus_depth = focus_depth
        self.unique_layers = unique_layers
        self.n_h = n_h
        self.ls = ls
        self.g = g


class Results_class:
    # Saves results for all depths, for one instantiation of the tissue.
    def __init__(self, parameters, intensity_profiles):
        if len(parameters.depths) != len(intensity_profiles):
            raise ValueError("Number of intensity profiles does not match number of depths of interest.")
        self.depths = parameters.depths                                         # This is a list
        self.intensity_profiles = intensity_profiles                             # This is a list, same length as depths
        self.dx_for_export = parameters.dx_for_export
        self.wavelength = parameters.wavelength
        self.max_FDFD_dx = parameters.max_FDFD_dx
        self.resolution_factor = parameters.resolution_factor
        self.FDFD_dz = parameters.FDFD_dz
        self.beam_radius = parameters.beam_radius
        self.focus_depth = parameters.focus_depth
        self.unique_layers = parameters.unique_layers
        self.n_h = parameters.n_h
        self.ls = parameters.ls
        self.g = parameters.g
