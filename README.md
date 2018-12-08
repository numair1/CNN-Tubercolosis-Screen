Code Attribution Breakdown:

For an implementation of UNet, we used code and pretrained weights provided  by: https://github.com/milesial/Pytorch-UNet as a starting point. We modified it as per our needs, including changing the implementation of the dice loss, stopping the image from being split into two squares, adding in ability to save best performing epochs, adding in data cleaning scripts, adding in learning rate decay.

For ./scripts/transfer.py, we used the code from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html and changed it to meet our requirements for the classification part of our network. 
