import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda image: image.convert('RGB')),
    transforms.ToTensor(),  # [C, H, W] + float [0, 1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def preprocess_image(image):
    """
    Preprocess the image for model input.
    """
    
    image = transform(image)  # [C, H, W]
    image = image.unsqueeze(0)  # [1, C, H, W] - batch dimension

    return image  # Tensor ready for model
