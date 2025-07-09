import os
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

def plot_mean_variance(path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    files = sorted([
        os.path.join(path, f) for f in os.listdir(path)
        if f.lower().endswith(('.jpg', '.png'))
    ])

    means_before, vars_before = [], []
    means_after, vars_after = [], []

    for file in files:
        img = Image.open(file).convert("RGB")
        img_tensor_orig = transforms.ToTensor()(img)
        img_tensor_norm = transform(img)

        means_before.append(img_tensor_orig.mean().item())
        vars_before.append(img_tensor_orig.var().item())

        means_after.append(img_tensor_norm.mean().item())
        vars_after.append(img_tensor_norm.var().item())

    x = list(range(1, len(files)+1))

    plt.figure(figsize=(10,5))
    plt.plot(x, means_before, label='Mean Before', marker='o')
    plt.plot(x, means_after, label='Mean After', marker='o')
    plt.title("Mean per Image")
    plt.xlabel("Image Index")
    plt.ylabel("Mean")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10,5))
    plt.plot(x, vars_before, label='Variance Before', marker='o')
    plt.plot(x, vars_after, label='Variance After', marker='o')
    plt.title("Variance per Image")
    plt.xlabel("Image Index")
    plt.ylabel("Variance")
    plt.legend()
    plt.grid(True)
    plt.show()
