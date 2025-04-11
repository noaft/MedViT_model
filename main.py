import os
import matplotlib.pyplot as plt
from PIL import Image
from image_process import preprocess_image  # Assuming this is the correct import path
from model_load import model  # Assuming this is the correct import path

labels_truth = ['Actinic keratoses and intraepithelial carcinoma', \
          'Basal cell carcinoma', \
          'Benign keratosis-like lesions', \
          'Dermatofibroma', \
          'Melanoma', \
          'Melanocytic nevi', \
          'Vascular lesions']

labels = ['akiec', \
          'bcc', \
          'bkl', \
          'df', \
          'mel', \
          'nv', \
          'vasc']

def image_show(folder_path):
    """
    Display an image from a given folder.
    
    Args:
        folder_path (str): Path to the folder containing the image.
        image_name (str): Name of the image file.
    """
    
    # Get the list of image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Check if there are any images in the folder
    if not image_files:
        print("No images found in the specified folder.")
        return
    
    # Display each image
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        img = Image.open(image_path)

        input = preprocess_image(img)  # Preprocess the image
        label = model(input)

        plt.imshow(img)
        plt.axis('off')  # Hide axes
        plt.title(label)  # Show the image name as title
        plt.show()