import argparse
import pickle

import torch
from torch import nn
import numpy as np
from scipy import linalg
from tqdm import tqdm

#from model import Generator
from swagan import Generator
from calc_inception import load_patched_inception_v3

from train_distributed import train_args

@torch.no_grad()
def extract_feature_from_samples(
    generator, inception, truncation, truncation_latent, batch_size, n_sample, device
):
    n_batch = n_sample // batch_size
    resid = n_sample - (n_batch * batch_size)
    batch_sizes = [batch_size] * n_batch + [resid]
    features = []

    for batch in tqdm(batch_sizes):
        latent = torch.randn(batch, 512, device=device)
        img, _ = g([latent], truncation=truncation, truncation_latent=truncation_latent)
        feat = inception(img)[0].view(img.shape[0], -1)
        features.append(feat.to("cpu"))

    features = torch.cat(features, 0)

    return features


def calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print("product of cov matrices is singular")
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f"Imaginary component {m}")

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace

    return fid

class fid_params:
    std__params = {
        "truncation": 0.7, # truncation trick factor
        "truncation_mean": 4096, # number of vectors to calculate mean for truncation
        "batch": 232, # batch size for feeding into Inception
        "n_sample": 30000, # number of samples to calculate FID
        "size": 256, # image size
        "inception": "/home/ubuntu/hdd/Datasets/car_tile_only_img_dataset_lmdb/inception_car_tile_only_img_dataset_lmdb.pkl", # path to the precomputed inception embeddings
        "ckpt": "/home/ubuntu/hdd/stylegan2-pytorch/checkpoint/110000.pt", # path to the model checkpoint
    }

    def __init__(self, config=None):
        if config is not None:
            self.config = config
        else:
            self.config = self.std__params
        
        # load the config dict into the class
        for key, value in self.config.items():
            setattr(self, key, value)

if __name__ == "__main__":
    device = "cuda"

    args = fid_params()

    ckpt = torch.load(args.ckpt)

    g = Generator(args.size, 512, 8).to(device)
    g.load_state_dict(ckpt["g_ema"])
    g = nn.DataParallel(g)
    g.eval()

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g.module.mean_latent(args.truncation_mean)

    else:
        mean_latent = None

    inception = nn.DataParallel(load_patched_inception_v3()).to(device)
    inception.eval()

    features = extract_feature_from_samples(
        g, inception, args.truncation, mean_latent, args.batch, args.n_sample, device
    ).numpy()
    print(f"extracted {features.shape[0]} features")

    sample_mean = np.mean(features, 0)
    sample_cov = np.cov(features, rowvar=False)

    with open(args.inception, "rb") as f:
        embeds = pickle.load(f)
        real_mean = embeds["mean"]
        real_cov = embeds["cov"]

    fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)

    print("fid:", fid)
