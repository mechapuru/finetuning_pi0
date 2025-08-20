import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Create the directory if it doesn't exist
output_dir = "raw_images_jpg"
os.makedirs(output_dir, exist_ok=True)

try:
    base_image = np.load("raw_images/base_image.npy")
    wrist_image = np.load("raw_images/wrist_image.npy")

    print(f"Base image shape: {base_image.shape}, dtype: {base_image.dtype}")
    print(f"Wrist image shape: {wrist_image.shape}, dtype: {wrist_image.dtype}")

    # Convert NumPy array to PIL Image and save as JPG
    def save_as_jpg(image_array, filename):
        # Ensure the image is in uint8 format and has 3 channels (RGB)
        if image_array.dtype != np.uint8:
            if np.issubdtype(image_array.dtype, np.floating):
                image_array = (255 * image_array).astype(np.uint8)
            else:
                image_array = image_array.astype(np.uint8)
        if len(image_array.shape) == 3 and image_array.shape[-1] != 3:
            raise ValueError(f"Image should have 3 channels, got {image_array.shape[-1]}")
        if len(image_array.shape) == 2:
            image_array = np.stack((image_array,)*3, axis=-1)
        image = Image.fromarray(image_array)
        image.save(os.path.join(output_dir, filename), "JPEG")
        print(f"Saved {filename} to {output_dir}")

    save_as_jpg(base_image, "base_image_1_2.jpg")
    save_as_jpg(wrist_image, "wrist_image_1_2.jpg")



except FileNotFoundError:
    print("Error: raw_images/base_image.npy or raw_images/wrist_image.npy not found. Please make sure you have run the code that generates these files.")
except ValueError as e:
    print(f"Error: {e}")
    print("Please check the format of the images.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
