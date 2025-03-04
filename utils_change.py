import torch

## transformation functions
def get_reference_grid(grid_size):
    """
    生成标准化参考网格，形状为 [H, W, 2]
    """
    H, W = grid_size
    y_grid, x_grid = torch.meshgrid(
        torch.linspace(-1, 1, H), 
        torch.linspace(-1, 1, W),
        indexing='ij'  # 确保 PyTorch 2.0 兼容性
    )
    return torch.stack([x_grid, y_grid], dim=-1)  # 形状 [H, W, 2]


def warp_images(images, ddfs, ref_grids=None):
    """
    使用 `grid_sample` 进行图像配准
    images: [B, H, W]
    ddfs: [B, 2, H, W]
    ref_grids: [H, W, 2] or None
    """
    B, C, H, W = images.shape if images.dim() == 4 else (images.shape[0], 1, *images.shape[1:])

    # 生成标准参考网格
    if ref_grids is None:
        ref_grids = get_reference_grid((H, W)).to(ddfs.device)  # [H, W, 2]

    # 扩展网格到 batch 维度，并进行形变
    warped_grids = ref_grids.unsqueeze(0).expand(B, -1, -1, -1) + ddfs.permute(0, 2, 3, 1)  # [B, H, W, 2]

    # 确保 `images` 形状为 [B, C, H, W]
    if images.dim() == 3:
        images = images.unsqueeze(1)  # 变成 [B, 1, H, W]

    # 使用 `grid_sample` 进行变换
    warped_images = torch.nn.functional.grid_sample(images, warped_grids, align_corners=False, mode='bilinear', padding_mode='border')

    return warped_images.squeeze(1)  # 返回 [B, H, W]


## loss functions
def square_difference(i1, i2):
    return torch.mean((i1 - i2) ** 2, dim=(1, 2))


def gradient_dx(fv):
    return (fv[:, 2:, 1:-1] - fv[:, :-2, 1:-1]) / 2


def gradient_dy(fv):
    return (fv[:, 1:-1, 2:] - fv[:, 1:-1, :-2]) / 2


def gradient_txy(txy, fn):
    return torch.stack([fn(txy[:, i, ...]) for i in [0, 1]], axis=3)


def gradient_norm(displacement, flag_l1=False):
    dtdx = gradient_txy(displacement, gradient_dx)
    dtdy = gradient_txy(displacement, gradient_dy)
    if flag_l1:
        norms = torch.abs(dtdx) + torch.abs(dtdy)
    else:
        norms = dtdx**2 + dtdy**2
    return torch.mean(norms, dim=(1, 2, 3))
