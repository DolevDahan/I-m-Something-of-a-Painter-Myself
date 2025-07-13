#Dolev Dahan
#Ronel Davidov
import os, random, torch, json, numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as T
from argparse import Namespace
import matplotlib.pyplot as plt

_IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
def is_image_file(name):
    return name.lower().endswith(_IMG_EXT)

def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir) or os.path.islink(dir), f"{dir} is not a valid directory"
    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                images.append(os.path.join(root, fname))
    return images[: min(max_dataset_size, len(images))]

def copyconf(default_opt, **kwargs):
    conf = Namespace(**vars(default_opt))
    for k, v in kwargs.items():
        setattr(conf, k, v)
    return conf

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _make_power_2(img, base=4, method=Image.BICUBIC):
    ow, oh = img.size
    h, w = int(round(oh / base) * base), int(round(ow / base) * base)
    return img if (h == oh and w == ow) else img.resize((w, h), method)

def get_transform(opt, method=Image.BICUBIC):
    transforms = []

    if "resize" in opt.preprocess:
        size = [opt.load_size, opt.load_size]
        transforms.append(T.Resize(size, method))

    transforms.append(T.Lambda(lambda img: _make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        transforms.append(T.RandomHorizontalFlip())

    transforms += [
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    return T.Compose(transforms)

class UnalignedDataset(data.Dataset):
    def __init__(self, opt):
        self.opt           = opt
        self.root          = opt.dataroot
        self.current_epoch = 0

        self.seed = getattr(opt, "seed", None)
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        self._rng = random.Random(self.seed) if self.seed is not None else random

        phase_suffix = "" if opt.phase.endswith(("A", "B")) else opt.phase
        self.dir_A = os.path.join(opt.dataroot, f"{phase_suffix}A")
        self.dir_B = os.path.join(opt.dataroot, f"{phase_suffix}B")

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.A_size  = len(self.A_paths)
        self.B_size  = len(self.B_paths)

        self.train_len = getattr(opt, "train_samples", None) if opt.phase == "train" else None
        self.base_transform = get_transform(opt)

    def __len__(self):
        if self.train_len is not None:
            return self.train_len
        return max(self.A_size, self.B_size)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]

        if self.opt.serial_batches:
            B_path = self.B_paths[index % self.B_size]
        else:
            B_path = self.B_paths[self._rng.randint(0, self.B_size - 1)]

        A_img = Image.open(A_path).convert("RGB")
        B_img = Image.open(B_path).convert("RGB")

        transform = self.base_transform

        return {
            "A":       transform(A_img),
            "B":       transform(B_img),
            "A_paths": A_path,
            "B_paths": B_path,
        }


class SingleDataset(data.Dataset):
    def __init__(self, opt):
        self.opt  = opt
        self.root = opt.dataroot
        assert os.path.isdir(self.root) or os.path.islink(self.root), \
            f"{self.root} is not a valid directory"

        self.A_paths = sorted(make_dataset(self.root, opt.max_dataset_size))
        self.A_size  = len(self.A_paths)

        self.transform = get_transform(opt)

        input_nc = opt.output_nc if getattr(opt, "direction", "AtoB") == "BtoA" else opt.input_nc
        if input_nc == 1:
            base_tf = self.transform.transforms[:-2]
            self.transform = T.Compose([
                *base_tf,
                T.Grayscale(num_output_channels=1),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,)),
            ])

    def __len__(self):
        return self.A_size

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        A_img  = Image.open(A_path).convert("RGB")
        A      = self.transform(A_img)
        B      = torch.zeros_like(A)

        return {
            "A": A, "B": B,
            "A_paths": A_path,
            "B_paths": A_path,
        }

class SimpleDataLoader:
    def __init__(self, dataloader, dataset, opt):
        self.dataloader = dataloader
        self.dataset    = dataset
        self.opt        = opt

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data

    def set_epoch(self, epoch):
        self.dataset.current_epoch = epoch





def plot_seeds_two_figures(histories):
    seeds = list(histories.keys())
    n_seeds = len(seeds)

    fig1, axs1 = plt.subplots(1, n_seeds, figsize=(5 * n_seeds, 4))
    if n_seeds == 1:
        axs1 = [axs1]

    for col, seed in enumerate(seeds):
        hist = histories[seed]
        epochs = list(range(1, len(hist['G']) + 1))
        axs1[col].plot(epochs, hist['G'], label='Generator Loss', color='blue', linewidth=2)
        axs1[col].plot(epochs, hist['D'], label='Discriminator Loss', color='red', linewidth=2)
        axs1[col].set_title(f'G vs D Loss (Seed {seed})')
        axs1[col].set_xlabel('Epoch')
        axs1[col].set_ylabel('Loss')
        axs1[col].legend()
        axs1[col].grid(True)

    plt.tight_layout()
    plt.show()

    fig2, axs2 = plt.subplots(1, n_seeds, figsize=(5 * n_seeds, 4))
    if n_seeds == 1:
        axs2 = [axs2]

    for col, seed in enumerate(seeds):
        hist = histories[seed]
        epochs = list(range(1, len(hist['GAN']) + 1))
        axs2[col].plot(epochs, hist['GAN'], label='GAN Loss', color='blue', linewidth=2)
        axs2[col].plot(epochs, hist['NCE'], label='NCE Loss', color='green', linewidth=2)
        axs2[col].set_title(f'GAN vs NCE Loss (Seed {seed})')
        axs2[col].set_xlabel('Epoch')
        axs2[col].set_ylabel('Loss')
        axs2[col].legend()
        axs2[col].grid(True)

    plt.tight_layout()
    plt.show()
        
def load_history(json_path):
    with open(json_path, 'r') as f:
        history = json.load(f)
    return history
