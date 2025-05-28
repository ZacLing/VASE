import torch

def apply_mask_flatten(value, mask):
    mask_out = mask.expand(-1, -1, value.size(-1)).flatten()
    out_flat = value.flatten()
    out_masked = out_flat[mask_out != 0]

    return out_masked