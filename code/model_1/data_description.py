import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np


def show_images(files, title_prefix):
    fig, axs = plt.subplots(1, 5, figsize=(15, 3))
    for idx, (ax, f) in enumerate(zip(axs, files[:5]), 1):
        img = Image.open(f)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"{title_prefix} {idx}", fontsize=10)
    plt.tight_layout()
    plt.show()



def compute_avg_rgb_histogram(image_paths, max_images=200):
    """Compute average RGB histogram over a sample of images."""
    hist_r = np.zeros(256)
    hist_g = np.zeros(256)
    hist_b = np.zeros(256)
    count = 0

    for path in image_paths[:max_images]:
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i, hist in enumerate([hist_r, hist_g, hist_b]):
            h, _ = np.histogram(img[:, :, i], bins=256, range=(0, 256))
            hist += h
        count += 1

    if count > 0:
        hist_r /= count
        hist_g /= count
        hist_b /= count

    return hist_r, hist_g, hist_b

def get_image_paths(folder):
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(('.jpg', '.png'))
    ]

def plot_histograms(hist_m, hist_p):
    x = np.arange(256)

    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    plt.title("Average RGB Histogram - Monet")
    plt.plot(x, hist_m[0], color='red', label='R')
    plt.plot(x, hist_m[1], color='green', label='G')
    plt.plot(x, hist_m[2], color='blue', label='B')
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.title("Average RGB Histogram - Photo")
    plt.plot(x, hist_p[0], color='red', label='R')
    plt.plot(x, hist_p[1], color='green', label='G')
    plt.plot(x, hist_p[2], color='blue', label='B')
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
