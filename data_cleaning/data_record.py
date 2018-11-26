import os
import json
# Training imgs
data_split={}
data_split["train"]=[]
data_split["val"]=[]
for im in os.listdir("./../datasets/segmentation_dataset/train/imgs"):
    data_split["train"].append(im)

for im in os.listdir("./../datasets/segmentation_dataset/val/imgs"):
    data_split["val"].append(im)

with open("./../datasets/segmentation_dataset/MASTER.txt","wb+") as outfile:
    json_str = json.dumps(data_split)
    outfile.write(json_str)
