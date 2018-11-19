from scipy import misc
import os

for im in os.listdir("./../datasets/MontgomerySet/CXR_png"):
    img = misc.imread("./../datasets/MontgomerySet/CXR_png/"+im)
    mask = misc.imread("./../datasets/MontgomerySet/ManualMask/leftMask/"+im)

    resized_im = misc.imresize(img, (512,512))
    resized_mask = misc.imresize(mask, (512,512))
    
    misc.imsave("./../datasets/MontgomerySet/resized_img/"+im,resized_im)
    misc.imsave("./../datasets/MontgomerySet/resized_mask/"+im,resized_mask)
