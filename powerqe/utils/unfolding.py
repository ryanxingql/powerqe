"""Copyright 2023 RyanXingQL.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at
https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
import torch.nn.functional as F


def cal_diff(sz, sz_mul):
    diff_sz = int(np.ceil(sz / sz_mul)) * sz_mul - sz
    return diff_sz


def pad_img_sz_mul(img, sz_mul):
    """Image padding.

    Args:
        img (Tensor): Image with the shape of (N, C, H, W).
        sz_mul (int): Height and width of the padded image should be divisible
            by this factor.

    Returns:
        Tensor: Padded image with the shape of (N, C, H, W).
        Tuple: Padding information recorded as (left, right, top, bottom).
    """
    h, w = img.shape[2:]
    if (h < sz_mul) or (w < sz_mul):
        raise ValueError(f'Height ({h}) and width ({w}) should not be smaller'
                         f' than the patch size ({sz_mul}).')
    diff_h, diff_w = cal_diff(h, sz_mul), cal_diff(w, sz_mul)
    pad_info = ((diff_w // 2), (diff_w - diff_w // 2), (diff_h // 2),
                (diff_h - diff_h // 2))
    img_pad = F.pad(img, pad_info, mode='reflect')
    return img_pad, pad_info


def pad_img_min_sz(img, minSize):
    """Image padding.

    Args:
        img (Tensor): Image with the shape of (N, C, H, W).
        minSize (int): Minimum height and width of the padded image.

    Returns:
        Tensor: Padded image with the shape of (N, C, H, W).
        Tuple: Padding information recorded as (left, right, top, bottom).
    """
    h, w = img.shape[2:]
    diff_h = minSize - h if h < minSize else 0
    diff_w = minSize - w if w < minSize else 0
    pad_info = ((diff_w // 2), (diff_w - diff_w // 2), (diff_h // 2),
                (diff_h - diff_h // 2))
    img_pad = F.pad(img, pad_info, mode='reflect')
    return img_pad, pad_info


def unfold_img(img, patch_sz):
    """Image unfolding.

    Ref: "https://pytorch.org/docs/stable/generated/torch.Tensor.unfold.html"

    Args:
        img (Tensor): Image with the shape of (N, C, H, W).
        patch_sz (int): Unfolding patch size.

    Returns:
        Tensor: Unfolded patches with the shape of (B*N1*N2, C, PS, PS),
            where N1 is the patch number for H;
            N2 is the patch number for W;
            PS is the patch size.
        Tuple: Information for folding recording (B, N1, N2, C, PS, PS),
            where N1 is the patch number for H;
            N2 is the patch number for W;
            PS is the patch size.
    """
    patches = img.unfold(2, patch_sz, patch_sz).unfold(3, patch_sz, patch_sz)
    # unfold h: b c num_patch_h w patch_sz
    # unfold w: b c num_patch_h num_patch_w patch_sz patch_sz

    patches = patches.permute(0, 2, 3, 1, 4, 5)
    unfold_shape = patches.size()  # for reverting
    # b num_patch_h num_patch_w c patch_sz patch_sz
    patches = patches.contiguous().view(-1, img.size(1), patch_sz, patch_sz)
    # b*num_patch_h*num_patch_w c patch_sz patch_sz
    return patches, unfold_shape


def combine_patches(patches, unfold_shape):
    """Patch combination.

    Args:
        patches (Tensor): Patches with the shape of (B*N1*N2 C PS PS),
            where N1 is the patch number for H;
            N2 is the patch number for W;
            PS is the patch size.
        unfold_shape (Tuple): Information for folding recording
            (B, N1, N2, C, PS, PS), where N1 is the patch number for H;
            N2 is the patch number for W;
            PS is the patch size.

    Returns:
        Tensor: Image with the shape of (N, C, H, W).
    """
    b, c = unfold_shape[0], unfold_shape[3]
    h_pad = unfold_shape[1] * unfold_shape[4]
    w_pad = unfold_shape[2] * unfold_shape[5]

    img = patches.view(
        unfold_shape)  # b num_patch_h num_patch_w c patch_sz patch_sz
    img = img.permute(
        0, 3, 1, 4, 2,
        5).contiguous()  # b c num_patch_h patch_sz num_patch_w patch_sz
    img = img.view(b, c, h_pad, w_pad)
    return img


def crop_img(img, pad_info):
    """Image cropping.

    Args:
        img (Tensor): Image with the shape of (N, C, H, W).
        pad_info (Tuple): Padding information recorded as
            (left, right, top, bottom).

    Returns:
        Tensor: Cropped image.
    """
    if pad_info[3] == 0 and pad_info[1] == 0:
        img = img[..., pad_info[2]:, pad_info[0]:]
    elif pad_info[3] == 0:
        img = img[..., pad_info[2]:, pad_info[0]:-pad_info[1]]
    elif pad_info[1] == 0:
        img = img[..., pad_info[2]:-pad_info[3], pad_info[0]:]
    else:
        img = img[..., pad_info[2]:-pad_info[3], pad_info[0]:-pad_info[1]]
    return img
