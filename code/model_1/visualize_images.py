import os
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

def visualize_histograms(path, num_images=3):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    files = sorted([
        os.path.join(path, f) for f in os.listdir(path)
        if f.lower().endswith(('.jpg', '.png'))
    ])[:num_images]

    fig, axs = plt.subplots(len(files), 2, figsize=(10, 4*len(files)))

    if len(files) == 1:
        axs = [axs]

    for idx, file in enumerate(files):
        img = Image.open(file)
        img_tensor_orig = transforms.ToTensor()(img)
        img_tensor_norm = transform(img)

        axs[idx][0].hist(img_tensor_orig.flatten().numpy(), bins=50, color='blue', alpha=0.7)
        axs[idx][0].set_title(f"Histogram before normalize image {idx+1}")
        axs[idx][0].set_xlabel("Pixel Value")
        axs[idx][0].set_ylabel("Frequency")
        axs[idx][0].grid(True)

        axs[idx][1].hist(img_tensor_norm.flatten().numpy(), bins=50, color='red', alpha=0.7)
        axs[idx][1].set_title(f"Histogram after normalize image {idx+1}")
        axs[idx][1].set_xlabel("Pixel Value")
        axs[idx][1].set_ylabel("Frequency")
        axs[idx][1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
