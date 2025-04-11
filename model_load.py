import torch
import timm

def load_model(model_path, device = 'cpu'):
    """
    Load a pre-trained MedViT model from the specified path.

    Args:
        model_path (str): Path to the pre-trained model file.
        device (torch.device): Device to load the model onto (CPU or GPU).

    Returns:
        MedViT: Loaded MedViT model.
    """
    model = timm.create_model('MedViT_small', num_classes = 7) 
    
    # Load the state dictionary
    state_dict = torch.load(model_path, map_location=device)
    
    # Load the state dictionary into the model
    model.load_state_dict(state_dict)
    
    # Move the model to the specified device
    model.to(device)
    
    return model

model = load_model('./model/best_model.pth')