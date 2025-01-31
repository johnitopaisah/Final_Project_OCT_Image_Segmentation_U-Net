import os
from PIL import Image

# Path to the directory containing images
image_folder = '/content/drive/MyDrive/unet-oct-dataset-files/datasets/trainning/mask/01_CABONS_OD_masks'  # Replace with your image path
output_folder = '/content/drive/MyDrive/unet-oct-dataset-files/resize_folder/trainning/mask/01_CABONS_OD_masks'  # Folder to save cropped images

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through each file in the directory
for filename in sorted(os.listdir(image_folder)):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Check for valid image file extensions
        image_path = os.path.join(image_folder, filename)  # Create full image path
        image = Image.open(image_path)  # Load the image

        # # Define the crop area (left, upper, right, lower)
        # left = 510
        # upper = 0
        # right = 1264 
        # lower = 475

        # # Crop the image
        # cropped_image = image.crop((left, upper, right, lower))

        # Resize the cropped image to 960 x 480
        resized_image = image.resize((960, 480), Image.LANCZOS)  # LANCZOS filter maintains quality

        # Define output image path
        output_image_path = os.path.join(output_folder, filename)  # Save with the same filename in the output folder

        # Save the resized image
        resized_image.save(output_image_path)

        print(f"Resized image saved to {output_image_path}")
