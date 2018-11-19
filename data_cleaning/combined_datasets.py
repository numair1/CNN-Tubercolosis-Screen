import os
import shutil


# Move masks and imgs from China Set to master folder
china_mask_lists = os.listdir("./../datasets/ChinaSet_AllFiles/mask/")
for img in os.listdir("./../datasets/ChinaSet_AllFiles/CXR_png"):
	if img.split(".png")[0]+"_mask.png" not in china_mask_lists:
		continue
	shutil.copy("./../datasets/ChinaSet_AllFiles/CXR_png/"+img, "./../prepared_datasets/master/unsplit_data/imgs/"+img)
	shutil.copy("./../datasets/ChinaSet_AllFiles/mask/"+img.split(".png")[0]+"_mask.png", "./../prepared_datasets/master/unsplit_data/masks/"+img)



# Move masks and imgs from Montgomery Set to master folder
for img in os.listdir("./../datasets/MontgomerySet/CXR_png"):
        shutil.copy("./../datasets/MontgomerySet/CXR_png/"+img, "./../prepared_datasets/master/unsplit_data/imgs/"+img)
        shutil.copy("./../datasets/MontgomerySet/combinedMask/combined_"+img, "./../prepared_datasets/master/unsplit_data/masks/combined_"+img)
 
