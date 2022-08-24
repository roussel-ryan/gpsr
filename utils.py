import torch


def get_slopes(px, py, pz):
    denom = torch.sqrt((1.0 + pz) ** 2 - px ** 2 - py ** 2)
    xp = px / denom
    yp = py / denom
    return xp, yp


def calc_rms_size(xx, images):
    # note image sums are normalized to 1
    xx_ = xx[0].unsqueeze(0).unsqueeze(0).repeat(*images.shape[:2], 1, 1).to(images)
    yy_ = xx[1].unsqueeze(0).unsqueeze(0).repeat(*images.shape[:2], 1, 1).to(images)
    proj_x = images.sum(dim=-1)
    proj_y = images.sum(dim=-2)

    x = xx_[..., :, 0]
    y = yy_[..., 0, :]

    mean_x = torch.sum(x * proj_x, dim=-1).unsqueeze(-1)
    mean_y = torch.sum(y * proj_y, dim=-1).unsqueeze(-1)

    var_x = torch.sum((x - mean_x) ** 2 * proj_x, dim=-1).unsqueeze(-1)
    var_y = torch.sum((y - mean_y) ** 2 * proj_y, dim=-1).unsqueeze(-1)

    return torch.cat([mean_x, mean_y], dim=-1), torch.cat([var_x, var_y], dim=-1)
