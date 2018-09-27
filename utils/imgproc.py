import numpy as np
import torch


def remove_small_mask(mask):
    if mask.sum() < 15:
        return np.zeros(mask.shape)
    else:
        return mask


def remove_small_mask_batch(mask):

    batch_size = mask.shape[0]
    mask_reshape = torch.reshape(mask, (batch_size, -1))

    valid = mask_reshape.sum(dim=1) >= 100
    valid = torch.reshape(valid, [-1, 1, 1, 1])
    new_mask = mask * valid

    return new_mask
