import argparse
from io import BytesIO
import multiprocessing
from functools import partial

from PIL import Image
import lmdb
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import functional as trans_fn


def resize_and_convert(img, size, resample, quality=100):
    img = trans_fn.resize(img, size, resample)
    img = trans_fn.center_crop(img, size)
    buffer = BytesIO()
    img.save(buffer, format="jpeg", quality=quality)
    val = buffer.getvalue()

    return val


def resize_multiple(
    img, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS, quality=100
):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, size, resample, quality))

    return imgs


def resize_worker(img_file, sizes, resample):
    i, file = img_file
    img = Image.open(file)
    img = img.convert("RGB")
    out = resize_multiple(img, sizes=sizes, resample=resample)

    return i, out


def prepare(
    env, dataset, n_worker, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS
):
    resize_fn = partial(resize_worker, sizes=sizes, resample=resample)

    files = sorted(dataset.imgs, key=lambda x: x[0])
    files = [(i, file) for i, (file, label) in enumerate(files)]
    total = 0

    with multiprocessing.Pool(n_worker) as pool:
        for i, imgs in tqdm(pool.imap_unordered(resize_fn, files)):
            for size, img in zip(sizes, imgs):
                key = f"{size}-{str(i).zfill(5)}".encode("utf-8")

                with env.begin(write=True) as txn:
                    txn.put(key, img)

            total += 1

        with env.begin(write=True) as txn:
            txn.put("length".encode("utf-8"), str(total).encode("utf-8"))

class prepare_args:
    std_conf = {
        "path": "/home/ubuntu/hdd/Datasets/tile_dataset_1_defect_masks/Bruise", # path to the imgs folder
        "out_path": "/home/ubuntu/hdd/Datasets/tile_dataset_1_defect_masks_bruises_lmdb", # path to the output folder (lmdb dataset)
        "n_worker": 8, # number of workers
        "size": [128, 256], # sizes of the images
        "resample": "lanczos", # resample method
    }

    def __init__(self, config=None):
        if config is None:
            config = self.std_conf
        else:
            self.config = config

        # set the parameters
        for key, value in config.items():
            setattr(self, key, value)


if __name__ == "__main__":

    args = prepare_args()

    resample_map = {"lanczos": Image.LANCZOS, "bilinear": Image.BILINEAR}
    resample = resample_map[args.resample]

    sizes = args.size #[int(s.strip()) for s in args.size.split(",")]

    print("Make dataset of image sizes:", ", ".join(str(s) for s in sizes))

    imgset = datasets.ImageFolder(args.path)

    with lmdb.open(args.out_path, map_size=1024 ** 4, readahead=False) as env:
        prepare(env, imgset, args.n_worker, sizes=sizes, resample=resample)
