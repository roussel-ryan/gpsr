import torch


def get_slopes(px, py, pz):
    denom = torch.sqrt((1.0 + pz) ** 2 - px ** 2 - py ** 2)
    xp = px / denom
    yp = py / denom
    return xp, yp
