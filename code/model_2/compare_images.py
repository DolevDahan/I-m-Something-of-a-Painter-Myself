import os
import matplotlib.pyplot as plt
from PIL import Image
import random

def show_comparison(photo_dir, model1_dir, model2_dir, filename):
    """
    Displays a comparison between the original photo and the generated images
    from two different models.

    Parameters:
        photo_dir (str): Directory containing original photos.
        model1_dir (str): Directory containing generated images from Model 1.
        model2_dir (str): Directory containing generated images from Model 2.
        filename (str): Name of the image file to display.
    """
    photo_path = os.path.join(photo_dir, filename)
    model1_path = os.path.join(model1_dir, filename)
    model2_path = os.path.join(model2_dir, filename)

    photo_img = Image.open(photo_path)
    model1_img = Image.open(model1_path)
    model2_img = Image.open(model2_path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for i in range(2):
        axes[0, i].imshow(photo_img)
        axes[0, i].set_title(f"Orignal Photo")
        axes[0, i].axis('off')

    axes[1, 0].imshow(model1_img)
    axes[1, 0].set_title("Generated Model 1")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(model2_img)
    axes[1, 1].set_title("Generated Model 2")
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()



# def show_random_comparison(photo_dir, model1_dir, model2_dir, filenames):
#     """
#     Displays comparisons between the original photo and the generated images
#     from two different models for a list of random filenames.

#     Parameters:
#         photo_dir (str): Directory containing original photos.
#         model1_dir (str): Directory containing generated images from Model 1.
#         model2_dir (str): Directory containing generated images from Model 2.
#         filenames (list): List of filenames to display.
#     """
#     for filename in filenames:
#         photo_path = os.path.join(photo_dir, filename)
#         model1_path = os.path.join(model1_dir, filename)
#         model2_path = os.path.join(model2_dir, filename)

#         photo_img = Image.open(photo_path)
#         model1_img = Image.open(model1_path)
#         model2_img = Image.open(model2_path)

#         fig, axes = plt.subplots(2, 2, figsize=(8, 6))

#         for i in range(2):
#             axes[0, i].imshow(photo_img)
#             axes[0, i].set_title(f"Orignal Image")
#             axes[0, i].axis('off')

#         axes[1, 0].imshow(model1_img)
#         axes[1, 0].set_title("Generated Model 1")
#         axes[1, 0].axis('off')

#         axes[1, 1].imshow(model2_img)
#         axes[1, 1].set_title("Generated Model 2")
#         axes[1, 1].axis('off')

#         plt.tight_layout()
#         plt.show()


def show_random_comparison(photo_dir, model1_dir, model2_dir, filenames):
    """
    Displays all comparisons in a single figure.
    Each image gets a single row: Original | Model 1 | Model 2
    """
    n = len(filenames)
    fig, axes = plt.subplots(n, 3, figsize=(12, 3 * n))

    if n == 1:  
        axes = [axes]

    for idx, filename in enumerate(filenames):
        photo_path = os.path.join(photo_dir, filename)
        model1_path = os.path.join(model1_dir, filename)
        model2_path = os.path.join(model2_dir, filename)

        photo_img = Image.open(photo_path)
        model1_img = Image.open(model1_path)
        model2_img = Image.open(model2_path)

        axes[idx][0].imshow(photo_img)
        axes[idx][0].set_title("Original")
        axes[idx][0].axis('off')

        axes[idx][1].imshow(model1_img)
        axes[idx][1].set_title("Generated Model 1")
        axes[idx][1].axis('off')

        axes[idx][2].imshow(model2_img)
        axes[idx][2].set_title("Generated Model 2")
        axes[idx][2].axis('off')

    plt.tight_layout()
    plt.show()

