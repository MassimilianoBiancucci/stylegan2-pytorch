import argparse

import torch
from torchvision import utils

#from model import Generator
from swagan import Generator

from tqdm import tqdm

from train_distributed import train_args


def generate(args, g_ema, device, mean_latent):

    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
            sample_z = torch.randn(args.sample, args.latent, device=device)

            sample, _ = g_ema(
                [sample_z], truncation=args.truncation, truncation_latent=mean_latent
            )

            utils.save_image(
                sample,
                f"sample/{str(i).zfill(6)}.png",
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )

class inference_params:
    std_params = {
        "size": 256, # image size
        "sample": 1, # number of samples to generate foreach pic
        "pics": 100, # number of pictures
        "latent": 512, # latent vector size
        "n_mlp": 8, # number of MLP layers
        "truncation": 15.0, # truncation trick factor
        "truncation_mean": 4096, # number of vectors to calculate mean for truncation
        "ckpt": "/home/max/thesis/stylegan2-pytorch/checkpoint/car_tile_gan_110000.pt", # path to the model checkpoint
        "channel_multiplier": 2, # channel multiplier used during the training
    }

    def __init__(self, config=None):
        if config is not None:
            self.config = config
        else:
            self.config = self.std_params
        
        # load the config dict into the class
        for key, value in self.config.items():
            setattr(self, key, value)


if __name__ == "__main__":
    device = "cuda"

    args = inference_params()

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint["g_ema"])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, g_ema, device, mean_latent)
