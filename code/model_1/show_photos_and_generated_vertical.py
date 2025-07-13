#Dolev Dahan
#Ronel Davidov
import os
import matplotlib.pyplot as plt
from PIL import Image


def show_photos_and_generated_vertical(photo_dir, gen_dir, filenames=None):
    """
    Displays 3 pairs of original & generated images vertically.
    Each column is a photo & its generated version.

    :param photo_dir: Path to folder with original photos
    :param gen_dir: Path to folder with generated images
    :param filenames: Optional list of filenames (without path)
    """
    if filenames is None:
        filenames = sorted(os.listdir(photo_dir))[:3]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.tight_layout(pad=3)

    for idx, fname in enumerate(filenames):
        photo_path = os.path.join(photo_dir, fname)
        gen_path = os.path.join(gen_dir, fname)

        img_photo = Image.open(photo_path)
        img_gen = Image.open(gen_path)

        axes[0, idx].imshow(img_photo)
        axes[0, idx].set_title(f"Photo {idx + 1}")
        axes[0, idx].axis('off')

        axes[1, idx].imshow(img_gen)
        axes[1, idx].set_title(f"Generated {idx + 1}")
        axes[1, idx].axis('off')

    plt.show()


