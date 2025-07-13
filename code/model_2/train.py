#Dolev Dahan
#Ronel Davidov
import sys, time, json, random, os
from pathlib import Path
from math import inf
from pprint import pprint

import numpy as np
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt  # נדרש ל-matplotlib גם אם לא מציירים בפועל

PROJECT_DIR = Path(__file__).resolve().parent / "I-m-Something-of-a-Painter-Myself" / "code" / "model_2"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from dataset              import UnalignedDataset, SimpleDataLoader
from main_model_networks  import build_model


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def train(opt):
    set_global_seed(opt.seed)
    g = torch.Generator().manual_seed(opt.seed)
    dataset_inst = UnalignedDataset(opt)
    print(f"dataset [{type(dataset_inst).__name__}] was created")

    loader = data.DataLoader(
        dataset_inst,
        batch_size   = opt.batch_size,
        shuffle      = not opt.serial_batches,
        num_workers  = int(opt.num_threads),
        drop_last    = bool(opt.isTrain),
        worker_init_fn = seed_worker,
        generator    = g,
        pin_memory   = True,
    )

    dataset = SimpleDataLoader(loader, dataset_inst, opt)
    dataset_size = len(dataset)
    print(f"The number of training images = {dataset_size}")


    model = build_model(opt)

    hist = {'D': [], 'G': [], 'GAN': [], 'NCE': []}
    total_iters = 0

    print('----OPT----')
    pprint(vars(opt))
    print('----OPT----\n')

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        epoch_iter       = 0
        model.init_epoch_stats()
        dataset.set_epoch(epoch)

        for i, data_i in enumerate(dataset):
            if opt.gpu_ids:
                torch.cuda.synchronize()

            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data_i)
                model.setup(opt)

            model.set_input(data_i)
            model.optimize_parameters()

            losses = model.get_current_losses()
            batch_size = data_i["A"].size(0)
            model.accumulate_epoch_stats(losses, batch_size)

            total_iters += batch_size
            epoch_iter  += batch_size


            if total_iters % opt.save_latest_freq == 0:
                tag = 'latest' if not opt.save_by_iter else f'iter_{total_iters}'
                model.save_networks(tag)

        model.print_epoch_stats(epoch)
        es = model.epoch_stats
        n  = max(1, es['samples'])
        for k in ('D', 'G', 'GAN', 'NCE'):
            hist[k].append(es[k] / n)

        if epoch % opt.save_epoch_freq == 0:
            model.save_networks('latest')
            model.save_networks(epoch)

        model.update_learning_rate()
        print(f"[Epoch {epoch:03d}] time={time.time() - epoch_start_time:.1f}s")

    # 2.6  שמירת היסטוריה מלאה
    hist_path = Path(opt.checkpoints_dir) / f"{opt.name}_hist_seed_{opt.seed}.json"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hist_path, "w") as f:
        json.dump(hist, f, indent=2)
    print(f"[INFO] saved full loss history → {hist_path.resolve()}")
