import torch
from Vanila_models import GeneratorResNet
from torchvision import transforms
from PIL import Image
import os

def generate_images(checkpoint_path, input_dir, output_dir, batch_size=5, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_dir, exist_ok=True)

    G_AB = GeneratorResNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    G_AB.load_state_dict(checkpoint['G_AB'])
    G_AB.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    to_pil = transforms.ToPILImage()

    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))]

    print(f"Found {len(files)} images in {input_dir}.")

    count = 0
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i+batch_size]
        imgs = []
        for f in batch_files:
            img = Image.open(f)
            img = transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs, 0).to(device)

        with torch.no_grad():
            fake_imgs = G_AB(imgs).cpu()
            fake_imgs = (fake_imgs + 1) / 2

        for j, f in enumerate(batch_files):
            out_img = to_pil(fake_imgs[j])
            name = os.path.basename(f)
            out_path = os.path.join(output_dir, name)
            out_img.save(out_path)
            count += 1

    print(f"{count} generated images were saved to {output_dir}")
