import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

from unet import UNet
from utils import resize_and_crop, normalize, split_img_into_squares, hwc_to_chw
from utils import plot_img_and_mask

from torchvision import transforms

def predict_img(net,
                full_img,
                scale_factor=0.5,
                out_threshold=0.5,
                use_dense_crf=True,
                use_gpu=False):

    net.eval()
    img_height = full_img.size[1]
    img_width = full_img.size[0]

    img = resize_and_crop(full_img, scale=scale_factor)
    img = normalize(img)

    img = hwc_to_chw(img)

    X = torch.from_numpy(img).unsqueeze(0)
    
    if use_gpu:
        X = X.cuda()

    with torch.no_grad():
        output = net(X)

        probs = F.sigmoid(output).squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(img_height),
                transforms.ToTensor()
            ]
        )
        
        probs = tf(probs.cpu())

        mask_np = probs.squeeze().cpu().numpy()

    full_mask = mask_np #merge_masks(left_mask_np, right_mask_np, img_width)

    if use_dense_crf:
        full_mask = dense_crf(np.array(full_img).astype(np.uint8), full_mask)

    return full_mask > out_threshold



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='filenames of ouput images')
    parser.add_argument('--cpu', '-c', action='store_true',
                        help="Do not use the cuda version of the net",
                        default=False)
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--no-crf', '-r', action='store_true',
                        help="Do not use dense CRF postprocessing",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()

def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        print("Error : Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

if __name__ == "__main__":
    args = get_args()
    in_file_dir = args.input[0]
    out_file_dir = get_output_filenames(args)[0]
    net = UNet(n_channels=3, n_classes=1)

    print("Loading model {}".format(args.model))

    if not args.cpu:
        print("Using CUDA version of the net, prepare your GPU !")
        net.cuda()
        net.load_state_dict(torch.load(args.model))
    else:
        net.cpu()
        net.load_state_dict(torch.load(args.model, map_location='cpu'))
        print("Using CPU version of the net, this may be very slow")

    print("Model loaded !")

    for im in os.listdir(in_file_dir):
        print("\nPredicting image {} ...".format(im))

        img = Image.open(in_file_dir+"/"+im)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           use_dense_crf= not args.no_crf,
                           use_gpu=not args.cpu)

        result = mask_to_image(mask)
        result.save(out_file_dir+"/"+im)
        print("Mask saved to {}".format(out_file_dir+"/"+im))
