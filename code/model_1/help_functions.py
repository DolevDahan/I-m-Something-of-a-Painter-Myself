#Dolev Dahan
#Ronel Davidov
import json
import matplotlib.pyplot as plt

def load_history(json_path):
    with open(json_path, 'r') as f:
        history = json.load(f)
    return history


def plot_losses(history):
    seed = history.get("seed", "unknown")
    epochs = list(range(1, len(history["G_losses"]) + 1))

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["G_losses"], label='Generator Loss')
    plt.plot(epochs, history["D_losses"], label='Discriminator Loss')
    plt.title(f'Generator vs Discriminator Loss (Seed: {seed})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["G_GAN_losses"], label='GAN Loss')
    plt.plot(epochs, history["G_cycle_losses"], label='Cycle Loss')
    plt.plot(epochs, history["G_identity_losses"], label='Identity Loss')
    plt.title(f'Generator Components Losses (Seed: {seed})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
