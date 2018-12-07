Code Attribution Breakdown:

For an implementation of UNet, we used code and pretrained weights provided  by: https://github.com/milesial/Pytorch-UNet as a starting point. We modified it as per our needs, including changing the implementation of the dice loss, stooping the image from being split into two squares, adding in features like saving best performing epochs, adding in data cleaning scripts.

For ./scripts/transfer.py, we used the code from ___ and changed it ___ . 

Data Cleaning:
1. Montgomery Set - Add the left lung and right lung masks together to create one mask file
2. China Set, Montogomery Set - Resize both masks and image to multiple of 32
3. Use transposed convolution instead biliniear upsampling for UNet
4. Parse out tubercolosis diagnosis from .txt files for both sets
