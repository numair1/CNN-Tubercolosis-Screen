from scipy import misc
import os
import numpy as np
def modify(name):
	return name.split('.')[0] + '_mask.png'

k = 1
gt_dir = './data/ChinaSet_AllFiles/mask/'
for x in ['train/', 'valid/']:
	d = 0
	pred_dir = './data/pred_mask/' + x
	seg_names = os.listdir(pred_dir)
	for mask in seg_names:
		if 'MCUCXR' in mask:
			continue
		#print(mask)
		gt = misc.imread(gt_dir + modify(mask))
		seg = misc.imread(pred_dir + mask)
		gt = misc.imresize(gt, seg.shape)
		dice = np.sum(seg[gt==k])*2.0 / (np.sum(seg) + np.sum(gt))
		#print(dice)
		d += dice
	print(x, d)
