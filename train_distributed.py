import argparse
import math
import random
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import transforms, utils
from tqdm import tqdm

try:
    import wandb

except ImportError:
    wandb = None


from dataset import MultiResolutionDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from op import conv2d_gradfix
from non_leaking import augment, AdaptiveAugment

# chose the model architecture
ARCH = "swgan"    
if ARCH == "stylegan2":
    from model import Generator, Discriminator # stylegan2
elif ARCH == "swgan":
    from swagan import Generator, Discriminator # swgan

### distributed functions
def ddp_setup(rank, world_size):
    """
    Setup distributed training.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    synchronize()


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):

    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )

    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )

    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device):
    
    loader = sample_data(loader)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 8, device)

    sample_z = torch.randn(args.n_sample, args.latent, device=device)

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")

            break

        real_img = next(loader)
        real_img = real_img.to(device)

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ = generator(noise)

        if args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)

        else:
            real_img_aug = real_img

        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img_aug)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        if args.augment and args.augment_p == 0:
            ada_aug_p = ada_augment.tune(real_pred)
            r_t_stat = ada_augment.r_t_stat

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True

            if args.augment:
                real_img_aug, _ = augment(real_img, ada_aug_p)

            else:
                real_img_aug = real_img

            real_pred = discriminator(real_img_aug)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ = generator(noise)

        if args.augment:
            fake_img, _ = augment(fake_img, ada_aug_p)

        fake_pred = discriminator(fake_img)
        g_loss = g_nonsaturating_loss(fake_pred)

        loss_dict["g"] = g_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0

        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_noise(path_batch_size, args.latent, args.mixing, device)
            fake_img, latents = generator(noise, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    f"augment: {ada_aug_p:.4f}"
                )
            )

            if i % 100 == 0:
                with torch.no_grad():
                    g_ema.eval()
                    sample, _ = g_ema([sample_z])
                    grid_sample = utils.make_grid(
                        sample, 
                        nrow=int(args.n_sample ** 0.5), 
                        normalize=True, 
                        value_range=(-1, 1)
                    )
                    utils.save_image(
                        grid_sample,
                        f"sample/{str(i).zfill(6)}.png"
                    )
                    wandb.log({})

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Samples": wandb.Image(grid_sample),
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Augment": ada_aug_p,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                    }
                )

            if i % 10000 == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(), 
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                        "ada_aug_p": ada_aug_p,
                    },
                    f"checkpoint/{str(i).zfill(6)}.pt",
                )


class train_args:
    """
    class that contain the same variables
    as the arguments of the train.py file
    """
    std_config = {
        "path": "/home/ubuntu/hdd/Datasets/car_tile_only_img_dataset_lmdb",
        "arch": ARCH,
        "iter": 800000,
        "batch": 32,
        "n_sample": 64,
        "size": 256,
        "r1": 10,
        "path_regularize": 2,
        "path_batch_shrink": 2,
        "d_reg_every": 16,
        "g_reg_every": 4,
        "mixing": 0.9,
        "ckpt": None,
        "lr": 0.002,
        "channel_multiplier": 2,
        
        "augment": False,
        "augment_p": 0.0,
        "ada_target": 0.6,
        "ada_length": 500*1000,
        "ada_every": 256,

        #"distributed": True,
        #"ngpus": 1, # number of gpus # unused overwritten by torch.cuda.device_count()
        #"local_rank": 0, # local rank for distributed training, default 0 for the principal machine

        "wandb": True, # set to false if you don't want to use wandb
        "wandb_project": "stylegan2",
        "wandb_entity": "deep_learning_team",
        "wandb_mode": "online",
    }

    def __init__(self, config=None):
        
        self.config = self.std_config
        if config is not None:
            self.config.update(config)
            
        
        self.path = self.config["path"]  # path to the lmdb dataset
        self.arch = self.config["arch"]  # model architectures (stylegan2 | swagan)
        self.iter = self.config["iter"]  # total training iterations, defult 800'000
        self.batch = self.config["batch"]    # batch sizes for each gpus, default: 16
        self.n_sample = self.config["n_sample"]  # number of the samples generated during training, default 64
        self.size = self.config["size"]  # image sizes for the model, default is 256
        self.r1 = self.config["r1"]  # weight of the r1 regularization, default is 10
        self.path_regularize = self.config["path_regularize"] # weight of the path regularization, default is 2
        self.path_batch_shrink = self.config["path_batch_shrink"] # batch size reducing factor for the path length regularization (reduce memory consumption), default is 2
        self.d_reg_every = self.config["d_reg_every"] # interval of the applying r1 regularization (discriminator), default is 16
        self.g_reg_every = self.config["g_reg_every"] # interval of the applying path length regularization (generator), default is 4
        self.mixing = self.config["mixing"] # probability of latent code mixing, default is 0.9
        self.ckpt = self.config["ckpt"] # path to the checkpoint to resume training, default is None
        self.lr = self.config["lr"] # learning rate, default is 0.002
        self.channel_multiplier = self.config["channel_multiplier"] #channel multiplier factor for the model. config-f = 2, else = 1, default 2
        
        self.augment = self.config["augment"] # apply non leaking augmentation, default is False
        self.augment_p = self.config["augment_p"] # probability of applying augmentation. 0 = use adaptive augmentation, default is 0
        self.ada_target = self.config["ada_target"] # target augmentation probability for adaptive augmentation, default is 0.6
        self.ada_length = self.config["ada_length"] # target duraing to reach augmentation probability for adaptive augmentation, default is 50'000 (500*1000)
        self.ada_every = self.config["ada_every"] # probability update interval of the adaptive augmentation, default is 256
        
        self.ngpus = torch.cuda.device_count(), #self.config["ngpus"] # number of gpus used
        #self.local_rank = self.config["local_rank"] # unused # local rank for distributed training, default is 0

        self.wandb = self.config["wandb"] # use wandb for logging, default is False
        self.wandb_project = self.config["wandb_project"] # wandb project name, default is None
        self.wandb_entity = self.config["wandb_entity"] # wandb entity name, default is None
        self.wandb_mode = self.config["wandb_mode"] # wandb run name, default is None

        self.distributed = True # that script run only with distributed = True
        self.latent = 512 # latent vector size
        self.n_mlp = 8 # number of layers in the mapping network, default is 8
        self.start_iter = 0 # starting iteration for resuming training, default is 0


def main(rank: int, world_size: int, args: train_args):
    
    device = rank

    torch.cuda.set_device(device)
    ddp_setup(rank, world_size)

    #args = train_args()

    #if args.distributed:
    #    torch.cuda.set_device(args.local_rank)
    #    torch.distributed.init_process_group(
    #        backend="nccl", 
    #        #init_method="env://"
    #    )
    #    synchronize()

    #if args.arch in ["stylegan2", "swgan"]:
    #    if args.arch == 'stylegan2':
    #        
    #
    #    elif args.arch == 'swagan':
    #        from swagan import Generator, Discriminator
    #else:
    #    raise NotImplementedError(f"{args.arch} is not implemented")

    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)

    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)

    g_ema.eval()
    accumulate(g_ema, generator, 0) # ???

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )

    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    # loading checkpoint
    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

    # using a distributed cluster
    generator = nn.parallel.DistributedDataParallel(
        generator,
        device_ids=[rank],
        output_device=rank,
        broadcast_buffers=False,
    )

    discriminator = nn.parallel.DistributedDataParallel(
        discriminator,
        device_ids=[rank],
        output_device=rank,
        broadcast_buffers=False,
    )


    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = MultiResolutionDataset(args.path, transform, args.size)

    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=True),
        drop_last=True,
    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(
            project=args.wandb_project, 
            entity=args.wandb_entity,
            mode=args.wandb_mode
        )

        wandb.config.update(args.__dict__)

    train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device)


if __name__ == "__main__":
    
    ### launching the distributed training
    args = train_args()

    world_size = torch.cuda.device_count()

    mp.spawn(main, args=(world_size, args), nprocs=world_size)
