import os
import shutil
import random

# Set values for data_path, dataset_name, train_ratio, test_ratio, and valid

# path to folder containing all datasets
data_path = './data/'

# name of dataset to split
dataset_name = 'ChinaSet_AllFiles'

train_ratio = 0.7
test_ratio = 0.15
valid = True

path  = data_path + dataset_name + '/CXR_png/'
mask  = data_path + dataset_name + '/mask/'

if valid:
	valid_ratio = 1 - train_ratio - test_ratio

names = os.listdir(path)
mask_names = set(os.listdir(mask))
if os.path.exists(data_path + 'split_' + dataset_name + '/'):
	shutil.rmtree(data_path + 'split_' + dataset_name + '/')
os.makedirs(data_path + 'split_' + dataset_name + '/train/class_0/img/')
os.makedirs(data_path + 'split_' + dataset_name + '/train/class_0/mask/')
os.makedirs(data_path + 'split_' + dataset_name + '/train/class_1/img/')
os.makedirs(data_path + 'split_' + dataset_name + '/train/class_1/mask/')
if valid:
	#print(valid_ratio)
	os.makedirs(data_path + 'split_' + dataset_name + '/valid/class_0/img/')
	os.makedirs(data_path + 'split_' + dataset_name + '/valid/class_0/mask/')
	os.makedirs(data_path + 'split_' + dataset_name + '/valid/class_1/img/')
	os.makedirs(data_path + 'split_' + dataset_name + '/valid/class_1/mask/')
os.makedirs(data_path + 'split_' + dataset_name + '/test/class_0/img/')
os.makedirs(data_path + 'split_' + dataset_name + '/test/class_0/mask/')
os.makedirs(data_path + 'split_' + dataset_name + '/test/class_1/img/')
os.makedirs(data_path + 'split_' + dataset_name + '/test/class_1/mask/')
final_names_0 = []
final_names_1 = []


check_mask = True if dataset_name == 'ChinaSet_AllFiles' else False

def modify(name):
	if check_mask:
		return name.split('.')[0] + '_mask.png'
	else:
		return name

for name in names:
	if modify(name) in mask_names:
		if '0.png' in name:
			final_names_0.append(name)
		elif '1.png' in name:
			final_names_1.append(name)
random.shuffle(final_names_0)
random.shuffle(final_names_1)
l_0 = len(final_names_0)
l_1 = len(final_names_1)
train_names_0 = set(final_names_0[:int(train_ratio * l_0)])
train_names_1 = set(final_names_1[:int(train_ratio * l_1)])
if valid:
	test_names_0  = set(final_names_0[int(train_ratio * l_0):int(train_ratio * l_0) + int(test_ratio * l_0)])
	valid_names_0 = set(final_names_0[int(train_ratio * l_0) + int(test_ratio * l_0):])
	test_names_1  = set(final_names_1[int(train_ratio * l_1): int(train_ratio * l_1) + int(test_ratio * l_1)])
	valid_names_1 = set(final_names_1[int(train_ratio * l_1) + int(test_ratio  * l_1):])
else:
	test_names_0  = set(final_names_0[int(train_ratio * l_0):])
	test_names_1  = set(final_names_1[int(train_ratio * l_1):])


for name in final_names_0:
	if name in train_names_0:
		shutil.copy(path + name, data_path + 'split_' + dataset_name + '/train/class_0/img/' + name)
		shutil.copy(mask + modify(name), data_path + 'split_' + dataset_name + '/train/class_0/mask/' + name)
		#shutil.copy(right + name, data_path + 'split_' + dataset_name + '/train/class_0/right/' + name)
	elif name in test_names_0:
		shutil.copy(path + name, data_path + 'split_' + dataset_name + '/test/class_0/img/' + name)
		shutil.copy(mask + modify(name), data_path + 'split_' + dataset_name + '/test/class_0/mask/' + name)
		#shutil.copy(right + name, data_path + 'split_' + dataset_name + '/test/class_0/right/' + name)
	elif name in valid_names_0:
		shutil.copy(path + name, data_path + 'split_' + dataset_name + '/valid/class_0/img/' + name)
		shutil.copy(mask + modify(name), data_path + 'split_' + dataset_name + '/valid/class_0/mask/' + name)
		#shutil.copy(right + name, data_path + 'split_' + dataset_name + '/valid/class_0/right/' + name)

for name in final_names_1:
	if name in train_names_1:
		shutil.copy(path + name, data_path + 'split_' + dataset_name + '/train/class_1/img/' + name)
		shutil.copy(mask + modify(name), data_path + 'split_' + dataset_name + '/train/class_1/mask/' + name)
		#shutil.copy(right + name, data_path + 'split_' + dataset_name + '/train/class_1/right/' + name)
	elif name in test_names_1:
		shutil.copy(path + name, data_path + 'split_' + dataset_name + '/test/class_1/img/' + name)
		shutil.copy(mask + modify(name), data_path + 'split_' + dataset_name + '/test/class_1/mask/' + name)
		#shutil.copy(right + name, data_path + 'split_' + dataset_name + '/test/class_1/right/' + name)
	elif name in valid_names_1:
		shutil.copy(path + name, data_path + 'split_' + dataset_name + '/valid/class_1/img/' + name)
		shutil.copy(mask + modify(name), data_path + 'split_' + dataset_name + '/valid/class_1/mask/' + name)
		#shutil.copy(right + name, data_path + 'split_' + dataset_name + '/valid/class_1/right/' + name)

#print(int(train_ratio * l_0) + int(test_ratio * l_0) + int(valid_ratio * l_0))

