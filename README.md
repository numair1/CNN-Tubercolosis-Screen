# Using Deep Convolutional Networks to Screen for Tuberculosis from Chest X Rays

This is the code repository for my class project in Deep Learning taken at Johns Hopkins University. The code outlines a pipelines that chains together image segmentation, conditional random fields and Convolutaional Neural Networks to classify X-Rays for presence of Tubercolosis.
Code Attribution Breakdown:

For an implementation of UNet, we used code and pretrained weights provided  by: https://github.com/milesial/Pytorch-UNet as a starting point. We modified it as per our needs, including changing the implementation of the dice loss, stopping the image from being split into two squares, adding in ability to save best performing epochs, adding in data cleaning scripts, adding in learning rate decay.

For ./scripts/transfer.py, we used the code from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html and changed it to meet our requirements for the classification part of our network. 
