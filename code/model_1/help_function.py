import json
import matplotlib.pyplot as plt

def load_history(json_path):
    with open(json_path, 'r') as f:
        history = json.load(f)
    return history


def plot_losses(history):
    seed = history.get("seed", "unknown")
    epochs = list(range(1, len(history["G_losses"]) + 1))

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["G_losses"], label='Generator Loss',color='blue',linewidth=2)
    plt.plot(epochs, history["D_losses"], label='Discriminator Loss',color='red',linewidth=2)
    plt.title(f'Generator vs Discriminator Loss (Seed: {seed})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["G_GAN_losses"], label='GAN Loss',color='blue',linewidth=2)
    plt.plot(epochs, history["G_cycle_losses"], label='Cycle Loss',color='red',linewidth=2)
    plt.plot(epochs, history["G_identity_losses"], label='Identity Loss',color='green',linewidth=2)
    plt.title(f'Generator Components Losses (Seed: {seed})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_seeds_two_figures(histories):
    seeds = list(histories.keys())
    n_seeds = len(seeds)

    fig1, axs1 = plt.subplots(1, n_seeds, figsize=(5*n_seeds, 4))
    if n_seeds == 1:
        axs1 = [axs1]
    for col, seed in enumerate(seeds):
        history = histories[seed]
        epochs = list(range(1, len(history["G_losses"]) + 1))
        axs1[col].plot(epochs, history["G_losses"], label='Generator Loss', color='blue', linewidth=2)
        axs1[col].plot(epochs, history["D_losses"], label='Discriminator Loss', color='red', linewidth=2)
        axs1[col].set_title(f'G vs D Loss (Seed {seed})')
        axs1[col].set_xlabel('Epoch')
        axs1[col].set_ylabel('Loss')
        axs1[col].legend()
        axs1[col].grid(True)
    plt.tight_layout()
    plt.show()

    fig2, axs2 = plt.subplots(1, n_seeds, figsize=(5*n_seeds, 4))
    if n_seeds == 1:
        axs2 = [axs2]
    for col, seed in enumerate(seeds):
        history = histories[seed]
        epochs = list(range(1, len(history["G_GAN_losses"]) + 1))
        axs2[col].plot(epochs, history["G_GAN_losses"], label='GAN Loss', color='blue', linewidth=2)
        axs2[col].plot(epochs, history["G_cycle_losses"], label='Cycle Loss', color='red', linewidth=2)
        axs2[col].plot(epochs, history["G_identity_losses"], label='Identity Loss', color='green', linewidth=2)
        axs2[col].set_title(f'G Components Loss (Seed {seed})')
        axs2[col].set_xlabel('Epoch')
        axs2[col].set_ylabel('Loss')
        axs2[col].legend()
        axs2[col].grid(True)
    plt.tight_layout()
    plt.show()



