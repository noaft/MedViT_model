import torch
import timm
import MedViT

def load_model(model_path, device = 'cpu'):
    """
    Load a pre-trained MedViT model from the specified path.

    Args:
        model_path (str): Path to the pre-trained model file.
        device (torch.device): Device to load the model onto (CPU or GPU).

    Returns:
        MedViT: Loaded MedViT model.
    """
    # Create model using timm's create_model
    model = timm.create_model('MedViT_small', num_classes=7)
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model state dict from checkpoint
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)  # 'model' là key trong checkpoint
    
    # Move model to device
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    return model


# import torch
# from MedViT import MedViT_small  # Giả sử bạn đã clone repo và có file MedViT.py

# def load_model(model_path, device='cpu'):
#     """
#     Load a pre-trained MedViT model from checkpoint.

#     Args:
#         model_path (str): Path to the pre-trained model checkpoint (.pth).
#         device (str): Device to load the model onto.

#     Returns:
#         model (torch.nn.Module): Loaded model in eval mode.
#     """
#     # Khởi tạo model đúng kiến trúc
#     model = MedViT_small(num_classes=7)  # số lớp phù hợp với task của bạn

#     # Load checkpoint
#     checkpoint = torch.load(model_path, map_location=device)

#     # Load state_dict vào model
#     model.load_state_dict(checkpoint['model'])  # 'model' là key trong checkpoint

#     model.to(device)
#     model.eval()
#     return model



model = load_model('./model/checkpoint_best.pth')