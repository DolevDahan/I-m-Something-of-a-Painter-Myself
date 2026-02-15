import random
import numpy as np
import torch
import json
import os
from Vanila_models import GeneratorResNet, Discriminator
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def train_cycle_gan(seed, path_monet, path_photo, save_dir, batch_size=5, n_epochs=50):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset_A_full = datasets.ImageFolder(path_monet, transform=transform)
    dataset_B_full = datasets.ImageFolder(path_photo, transform=transform)

    indices_A = np.random.choice(len(dataset_A_full), 200, replace=False)
    indices_B = np.random.choice(len(dataset_B_full), 200, replace=False)

    dataset_A = Subset(dataset_A_full, indices_A)
    dataset_B = Subset(dataset_B_full, indices_B)

    loader_A = DataLoader(dataset_A, batch_size=batch_size, shuffle=True)
    loader_B = DataLoader(dataset_B, batch_size=batch_size, shuffle=True)

    print(f"Training for seed: {seed}")
    print(f"Total Monet images: {len(dataset_A_full)}")
    print(f"Total Photo images: {len(dataset_B_full)}")
    print(f"Monet images in training subset: {len(dataset_A)}")
    print(f"Photo images in training subset: {len(dataset_B)}")

    history = {
        "seed": seed,
        "G_losses": [],
        "D_losses": [],
        "G_GAN_losses": [],
        "G_cycle_losses": [],
        "G_identity_losses": [],
        "logs": []
    }

    G_AB = GeneratorResNet().to(device)
    G_BA = GeneratorResNet().to(device)
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)

    adversarial_loss = nn.MSELoss()
    cycle_loss = nn.L1Loss()
    identity_loss = nn.L1Loss()

    optimizer_G = optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(n_epochs):
        for i, (data_A, data_B) in enumerate(zip(loader_A, loader_B)):
            real_A = data_A[0].to(device)
            real_B = data_B[0].to(device)

            G_AB.train()
            G_BA.train()
            optimizer_G.zero_grad()

            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)

            loss_id_A = identity_loss(G_BA(real_A), real_A)
            loss_id_B = identity_loss(G_AB(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B) / 2

            pred_fake_B = D_B(fake_B)
            pred_fake_A = D_A(fake_A)
            valid_B = torch.ones_like(pred_fake_B).to(device)
            valid_A = torch.ones_like(pred_fake_A).to(device)

            loss_GAN_AB = adversarial_loss(pred_fake_B, valid_B)
            loss_GAN_BA = adversarial_loss(pred_fake_A, valid_A)
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            recov_A = G_BA(fake_B)
            recov_B = G_AB(fake_A)
            loss_cycle_A = cycle_loss(recov_A, real_A)
            loss_cycle_B = cycle_loss(recov_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            loss_G = loss_GAN + 5 * loss_cycle + 10 * loss_identity
            loss_G.backward()
            optimizer_G.step()

            optimizer_D_A.zero_grad()
            pred_real_A = D_A(real_A)
            pred_fake_A = D_A(fake_A.detach())
            fake_target = torch.zeros_like(pred_fake_A).to(device)
            real_target = torch.ones_like(pred_real_A).to(device)
            loss_D_A = (adversarial_loss(pred_real_A, real_target) + adversarial_loss(pred_fake_A, fake_target)) / 2
            loss_D_A.backward()
            optimizer_D_A.step()

            optimizer_D_B.zero_grad()
            pred_real_B = D_B(real_B)
            pred_fake_B = D_B(fake_B.detach())
            fake_target = torch.zeros_like(pred_fake_B).to(device)
            real_target = torch.ones_like(pred_real_B).to(device)
            loss_D_B = (adversarial_loss(pred_real_B, real_target) + adversarial_loss(pred_fake_B, fake_target)) / 2
            loss_D_B.backward()
            optimizer_D_B.step()

        loss_D = (loss_D_A + loss_D_B) / 2

        log_epoch = (
            f"[Epoch {epoch+1}/{n_epochs}]\n"
            f"G: {loss_G.item():.4f} (GAN: {loss_GAN.item():.4f}, Cycle: {loss_cycle.item():.4f}, Identity: {loss_identity.item():.4f})\n"
            f"D: {loss_D.item():.4f} (D_A: {loss_D_A.item():.4f}, D_B: {loss_D_B.item():.4f})"
        )
        print(log_epoch)
        history["logs"].append(log_epoch)
        history["G_losses"].append(loss_G.item())
        history["D_losses"].append(loss_D.item())
        history["G_GAN_losses"].append(loss_GAN.item())
        history["G_cycle_losses"].append(loss_cycle.item())
        history["G_identity_losses"].append(loss_identity.item())

    weights_path = os.path.join(save_dir, f"checkpoint_seed_{seed}.pth")
    torch.save({
        'G_AB': G_AB.state_dict(),
        'G_BA': G_BA.state_dict(),
        'D_A': D_A.state_dict(),
        'D_B': D_B.state_dict(),
        'optimizer_G': optimizer_G.state_dict(),
        'optimizer_D_A': optimizer_D_A.state_dict(),
        'optimizer_D_B': optimizer_D_B.state_dict()
    }, weights_path)

    history_path = os.path.join(save_dir, f"history_seed_{seed}.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)

    print(f"Training for seed {seed} finished. Weights saved to {weights_path}, losses saved to {history_path}")

    return history
