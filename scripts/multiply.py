import os
from scipy import misc
import numpy as np
import shutil


mask_dir = './data/MontgomerySet/mask/'
image_dir = './data/MontgomerySet/CXR_png/'
save_dir = './data/tmp/'
if os.path.exists(save_dir):
	shutil.rmtree(save_dir)
os.makedirs(save_dir)
mask_names = os.listdir(mask_dir)
for mask in mask_names:
	image = misc.imread(image_dir + mask)
	mask_arr = misc.imread(mask_dir + mask)
	product = np.multiply(image, mask_arr)
	misc.imsave(save_dir + mask, product)

