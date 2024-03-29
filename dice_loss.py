import torch
from torch.autograd import Function, Variable
import numpy as np
class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        input =  (torch.sigmoid(input) > 0.5).float()
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input*input) + torch.sum(target*target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return -1*t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output *-1* 2 * (target * self.union - 2*input * self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)
