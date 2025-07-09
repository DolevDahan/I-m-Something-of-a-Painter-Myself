import torch
from Vanila_models import GeneratorResNet, Discriminator

def load_models(checkpoint_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G_AB = GeneratorResNet().to(device)
    G_BA = GeneratorResNet().to(device)
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    G_AB.load_state_dict(checkpoint['G_AB'])
    G_BA.load_state_dict(checkpoint['G_BA'])
    D_A.load_state_dict(checkpoint['D_A'])
    D_B.load_state_dict(checkpoint['D_B'])

    optimizer_G = torch.optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()))
    optimizer_D_A = torch.optim.Adam(D_A.parameters())
    optimizer_D_B = torch.optim.Adam(D_B.parameters())

    optimizer_G.load_state_dict(checkpoint['optimizer_G'])
    optimizer_D_A.load_state_dict(checkpoint['optimizer_D_A'])
    optimizer_D_B.load_state_dict(checkpoint['optimizer_D_B'])

    G_AB.eval()
    G_BA.eval()
    D_A.eval()
    D_B.eval()

    print(f"Models and optimizers loaded from {checkpoint_path}")
    return {
        "G_AB": G_AB,
        "G_BA": G_BA,
        "D_A": D_A,
        "D_B": D_B,
        "optimizer_G": optimizer_G,
        "optimizer_D_A": optimizer_D_A,
        "optimizer_D_B": optimizer_D_B
    }
