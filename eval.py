import torch
import torch.nn.functional as F
import numpy as np


def eval_net(net, dataset, gpu=False):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    for i, b in enumerate(dataset):
        img = b[0]
        true_mask = np.divide(b[1],255)

        img = torch.from_numpy(img).unsqueeze(0)
        true_mask = torch.from_numpy(true_mask).unsqueeze(0)

        if gpu:
            img = img.cuda()
            true_mask = true_mask.cuda()

        mask_pred = net(img)[0]
        mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
        eps = 0.0001
        inter = torch.dot(mask_pred.view(-1), true_mask.view(-1))
        union = torch.sum(mask_pred*mask_pred) + torch.sum(true_mask*true_mask) + eps
        tot += (2 *inter.float() + eps) / union.float() 
    return tot / i
