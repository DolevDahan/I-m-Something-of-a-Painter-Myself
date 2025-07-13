#Dolev Dahan
#Ronel Davidov
import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

def generate_images(G_AB, input_dir, output_dir, batch_size=5):
    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    print(f'CUDA Available: {cuda}')

    files = [os.path.join(input_dir, name) for name in os.listdir(input_dir)]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    generate_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    to_image = transforms.ToPILImage()

    G_AB.eval()

    for i in range(0, len(files), batch_size):
        imgs = []
        for j in range(i, min(len(files), i + batch_size)):
            img = Image.open(files[j])
            img = generate_transforms(img)
            imgs.append(img)
        imgs = torch.stack(imgs, 0).type(Tensor)

        fake_imgs = G_AB(imgs).detach().cpu()

        # שמירה
        for j in range(fake_imgs.size(0)):
            img = fake_imgs[j].squeeze().permute(1, 2, 0)
            img_arr = img.numpy()
            img_arr = (img_arr - np.min(img_arr)) * 255 / (np.max(img_arr) - np.min(img_arr))
            img_arr = img_arr.astype(np.uint8)

            img = to_image(img_arr)
            _, name = os.path.split(files[i + j])
            img.save(os.path.join(output_dir, name))

    print(f"Generated images saved in {output_dir}")
