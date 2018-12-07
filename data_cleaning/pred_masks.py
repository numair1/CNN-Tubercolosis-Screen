import os

for train_im in os.listdir("./datasets/segmentation_dataset/train/imgs/"):
	os.system("python predict.py -m dice_checkpoints/CP_best.pth -i datasets/segmentation_dataset/train/imgs/"+train_im+" -o datasets/pred_mask_dataset_binary_15/train/mask_"+train_im+" --no-crf  -s 0.15")

for val_im in os.listdir("./datasets/segmentation_dataset/val/imgs/"):
        os.system("python predict.py -m dice_checkpoints/CP_best.pth -i datasets/segmentation_dataset/val/imgs/"+val_im+" -o datasets/pred_mask_dataset_binary_15/val/mask_"+train
_im+" --no-crf  -s 0.15")
