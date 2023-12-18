import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss, mean.

    Per https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/5 with
    improvements for alpha.
    :param bce_loss: Binary Cross Entropy loss, a torch tensor.
    :param targets: a torch tensor containing the ground truth, 0s and 1s.
    :param pos_weight: weight of the class indicated by 1, a float scalar.
    :param gamma: focal loss power parameter, a float scalar.
    """

    def __init__(self, pos_weight=0.5, gamma=2.0, reduction="none"):
        nn.Module.__init__(self)
        self.pos_weight = pos_weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        p_t = torch.exp(-bce_loss)

        # pos_weight if target = 1 and 1 - pos_weight if target = 0
        alpha_tensor = (1 - self.pos_weight) + targets * (2 * self.pos_weight - 1)

        f_loss = alpha_tensor * (1 - p_t) ** self.gamma * bce_loss
        if self.reduction == "none":
            return f_loss
        elif self.reduction == "mean":
            return f_loss.mean()
        elif self.reduction == "sum":
            return f_loss.sum()
