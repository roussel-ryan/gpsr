import torch

def calculate_centroid(images, x, y):
    x_projection = images.sum(dim=-2)
    y_projection = images.sum(dim=-1)

    # calculate weighted avg
    x_centroid = (x_projection*x).sum(-1) / (x_projection.sum(-1) + 1e-8)
    y_centroid = (y_projection*y).sum(-1) / (y_projection.sum(-1) + 1e-8)

    return torch.stack((x_centroid, y_centroid))