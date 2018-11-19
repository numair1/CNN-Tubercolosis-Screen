import os
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

for im in os.listdir("./../datasets/MontgomerySet/ManualMask/leftMask/"):
	left_mask = misc.imread("./../datasets/MontgomerySet/ManualMask/leftMask/"+im)
	right_mask = misc.imread("./../datasets/MontgomerySet/ManualMask/rightMask/"+im)
	combined_mask = np.add(left_mask,right_mask)
	misc.imsave("./../datasets/MontgomerySet/combinedMask/combined_"+im, combined_mask)

