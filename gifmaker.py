import imageio.v2 as imageio
import os
import numpy as np

filenames = os.listdir('Results')
depths = np.array([40e-6,35e-6,30e-6,25e-6,20e-6,15e-6,10e-6,5e-6])

for depth in depths:
    
    filename_template = 'LG_'+str("{:02d}".format(int(1e6*depth)))+'um'
    images = []
    for filename in filenames:
        if filename[:len(filename_template)]==filename_template:
            images.append(imageio.imread('Results/'+filename))
    imageio.mimsave('Results/gifs/'+filename_template+'.gif', images, duration=500, palettesize = 512, quantizer ='wu', loop = 0)
    
    filename_template = 'HG_'+str("{:02d}".format(int(1e6*depth)))+'um'
    images = []
    for filename in filenames:
        if filename[:len(filename_template)]==filename_template:
            images.append(imageio.imread('Results/'+filename))
    imageio.mimsave('Results/gifs/'+filename_template+'.gif', images, duration=500, palettesize = 512, quantizer ='wu', loop = 0)
        