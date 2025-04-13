import os
import matplotlib.pyplot as plt
from PIL import Image
from image_process import preprocess_image  # Assuming this is the correct import path
from model_load import model  # Assuming this is the correct import path
import pandas as pd
import torch

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

def image_show(folder_path, meta_name):
    """
    Display an image from a given folder.
    
    Args:
        folder_path (str): Path to the folder containing the image.
        image_name (str): Name of the image file.
    """
    try:
        # Get the list of image files in the folder
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Check if there are any images in the folder
        if not image_files:
            print("No images found in the specified folder.")
            return
        
        df = pd.read_csv( os.path.join(folder_path, meta_name) )

        acc = 0
        len_items = len(image_files)

        # Display each image
        for image_file in image_files:
            try:
                image_path = os.path.join(folder_path, image_file)
                img = Image.open(image_path)

                input = preprocess_image(img)  # Preprocess the image
                output = model(input)  # Get the model prediction
                _, predicted = torch.max(output, 1)
                predicted = predicted.item()
                label = labels_truth[predicted]  # Get the label from the model prediction

                result = df[df['image_id'] ==  image_file.split('.')[0]] # find rows

                is_correct = result['dx'].values[0] == labels[predicted]
                if is_correct:
                    acc += 1

                plt.imshow(img)
                plt.axis('off')  # Hide axes
                plt.title(f"Predicted: {label}\nCorrect: {is_correct}")  # Show prediction and correctness
                plt.pause(2)  # Display image for 2 seconds before moving to next
                plt.clf()  # Clear the current figure
            except KeyboardInterrupt:
                print("\nProgram interrupted by user. Exiting...")
                plt.close()
                return
            except Exception as e:
                print(f"Error processing image {image_file}: {str(e)}")
                continue
        
        plt.close()  # Close the figure window after all images are displayed
        print(f"Accuracy: {acc/len_items*100:.2f}%")
        print(f"Total images: {len_items}")
        print(f"Correct predictions: {acc}")
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
        plt.close()
        return
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        plt.close()
        return

if __name__ == "__main__":
    folder_path = './image'
    meta_name = 'ISIC2018_Task3_Test_GroundTruth.csv'
    
    image_show(folder_path, meta_name)
    # Example usage
    # image_show(folder_path, meta_name)