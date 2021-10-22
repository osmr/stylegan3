# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""
import argparse
import os
import pickle

import PIL.Image
import numpy as np
import torch
from matplotlib import pyplot as plt


def generate_images(G, out_dir=None, device='cuda', seed=None,
                    truncation_psi=0.7, noise_mode='const', projected_w=None):
    """
    Generate images using pretrained network pickle.
    """
    assert (projected_w is not None)

    # w = np.load(projected_w)['w']
    # w = torch.tensor(w, device=device)

    projected_w_dict = np.load(projected_w)
    w = torch.tensor(projected_w_dict["w"], device=device)

    assert w.shape[1:] == (G.num_ws, G.w_dim)
    img = G.synthesis(w, noise_mode=noise_mode)
    # img = G.synthesis(w, noise_mode=noise_mode, force_fp32=True)

    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(os.path.join(
            out_dir,
            os.path.splitext(os.path.basename(projected_w))[0]) + '.png')

    return img[0].cpu().numpy()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network-pkl', type=str, required=True)
    parser.add_argument('--projected-w', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=False)
    return parser.parse_args()


def main():
    args = parse_args()
    print('Loading networks from "%s"...' % args.network_pkl)
    device = 'cuda'
    with open(args.network_pkl, 'rb') as f:
        G = pickle.load(f)['G_ema'].requires_grad_(False).to(device)
    img = generate_images(G, out_dir=args.out_dir, projected_w=args.projected_w)
    if args.out_dir is None:
        plt.rcParams["figure.figsize"] = (4, 4)
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    main()
