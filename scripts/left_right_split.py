import os
from scipy import misc
import numpy as np
import shutil
import cv2

remove_montgomery = True
save_dir = './data/split_pred/'
if os.path.exists(save_dir):
	shutil.rmtree(save_dir)
os.makedirs(save_dir + 'train/left/')
os.makedirs(save_dir + 'train/right/')
os.makedirs(save_dir + 'valid/left/')
os.makedirs(save_dir + 'valid/right/')
for x in ['train/', 'valid/']:
	mask_dir = './data/pred_mask/' + x
	mask_names = os.listdir(mask_dir)
	for mask in mask_names:
		print(mask)
		if remove_montgomery and 'MCUCXR' in mask:
			continue
		mask_arr = misc.imread(mask_dir + mask)
		s = np.sum(mask_arr, axis=0)
		left_start = False
		left_end = False
		left_end_idx = -1
		right_start_idx = -1
		left_start_idx = -1
		for i in range(len(s)):
			col_sum = s[i]
			if col_sum:
				if not left_end and not left_start:
					left_start = True
				elif left_end:
					right_start_idx = i
					break
			else:
				#print(left_start)
				if left_start:
					left_end_idx = i-1
					if left_end_idx - left_start_idx < 500:
						left_end_idx = -1
						left_start_idx = -1
					else:
						left_end = True
					left_start = False
		mid = int((left_end_idx + right_start_idx) / 2)
		left_mask = mask_arr[:, :mid+1]
		right_mask = mask_arr[:, mid+1:]

		misc.imsave(save_dir + x + 'left/'  + mask, left_mask)
		misc.imsave(save_dir + x + 'right/' + mask, right_mask)