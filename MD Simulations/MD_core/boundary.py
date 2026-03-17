import torch

def wrap_positions(positions, box):

    return positions % box

def minimum_image(diff, box):

    return diff - box * torch.round(diff / box)