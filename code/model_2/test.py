#Dolev Dahan
#Ronel Davidov
from main_model_networks import build_model
import torch

def load_trained_generator(opt, ckpt_path, device='cpu'):
    """
    Load a trained generator model from a given checkpoint path.
    
    Parameters:
        opt (Namespace): Options used to build the model.
        ckpt_path (str): Path to the checkpoint file (.pth).
        device (str): Device to load the model on ('cpu' or 'cuda').

    Returns:
        torch.nn.Module: The loaded generator model in eval mode.
    """
    opt.isTrain = False
    opt.gpu_ids = []

    model = build_model(opt)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.netG.load_state_dict(state_dict)
    model.netG.eval()
    model.netG = model.netG.to(device)

    print(f"netG loaded from {ckpt_path}")
    return model.netG
