import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio
from albumentations import HorizontalFlip, VerticalFlip, Rotate

""" Create a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path):
    train_image_dir = os.path.join(path, "datasets", "trainning", "images")
    train_mask_dir = os.path.join(path, "datasets", "trainning", "mask")
    
    test_image_dir = os.path.join(path, "datasets", "test", "images")
    test_mask_dir = os.path.join(path, "datasets", "test", "mask")
    
    train_x, train_y = [], []
    test_x, test_y = [], []

    # Load training images and masks
    for subfolder in os.listdir(train_image_dir):
        image_files = sorted(glob(os.path.join(train_image_dir, subfolder, "*.png")))
        train_x.extend(image_files)
        
    for subfolder in os.listdir(train_mask_dir):
        mask_files = sorted(glob(os.path.join(train_mask_dir, subfolder, "*.png")))
        train_y.extend(mask_files)

    # Load testing images and masks
    for subfolder in os.listdir(test_image_dir):
        image_files = sorted(glob(os.path.join(test_image_dir, subfolder, "*.png")))
        test_x.extend(image_files)
        
    for subfolder in os.listdir(test_mask_dir):
        mask_files = sorted(glob(os.path.join(test_mask_dir, subfolder, "*.png")))
        test_y.extend(mask_files)

    return (train_x, train_y), (test_x, test_y)

# def load_data(path):
#     train_x = sorted(glob(os.path.join(path, "datasets", "training", "**", "images", "*.png"), recursive=True))
#     train_y = sorted(glob(os.path.join(path, "datasets", "training", "mask", "**", "*.png"), recursive=True))

#     test_x = sorted(glob(os.path.join(path, "datasets", "test", "images", "**", "*.png"), recursive=True))
#     test_y = sorted(glob(os.path.join(path, "datasets", "test", "mask", "**", "*.png"), recursive=True))

#     return (train_x, train_y), (test_x, test_y)

# def augment_data(images, masks, save_path, augment=True):
#     size = (1920, 480)

#     for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
#         """ Extracting the name """
#         name = x.split("/")[-1].split(".")[0]

#         """ Reading image and mask """
#         x = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
#         x = np.expand_dims(x, axis=-1)
#         y = imageio.mimread(y)[0]


#         if augment == True:
#             aug = HorizontalFlip(p=1.0)
#             augmented = aug(image=x, mask=y)
#             x1 = augmented["image"]
#             y1 = augmented["mask"]

#             aug = Rotate(limit=15, p=1.0)
#             augmented = aug(image=x, mask=y)
#             x2 = augmented["image"]
#             y2 = augmented["mask"]

#             aug = Rotate(limit=-15, p=1.0)
#             augmented = aug(image=x, mask=y)
#             x3 = augmented["image"]
#             y3 = augmented["mask"]

#             X = [x, x1, x2, x3]
#             Y = [y, y1, y2, y3]


#         else:
#             X = [x]
#             Y = [y]

#         index = 0
#         for i, m in zip(X, Y):
#             i = cv2.resize(i, size)
#             m = cv2.resize(m, size)

#             tmp_image_name = f"{name}_{index}.png"
#             tmp_mask_name = f"{name}_{index}.png"

#             image_path = os.path.join(save_path, "image", tmp_image_name)
#             mask_path = os.path.join(save_path, "mask", tmp_mask_name)

#             cv2.imwrite(image_path, i.squeeze(-1))
#             cv2.imwrite(mask_path, m)

#             index += 1
def augment_data(images, masks, save_path, augment=True):
    size = (1920, 480)

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        """ Extracting the name """
        name = x.split("/")[-1].split(".")[0]

        """ Reading image and mask """
        x = cv2.imread(x, cv2.IMREAD_GRAYSCALE)  # Grayscale image
        x = np.expand_dims(x, axis=-1)
        y = imageio.mimread(y)[0]

        if augment == True:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = Rotate(limit=15, p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            aug = Rotate(limit=-10, p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented["image"]
            y3 = augmented["mask"]

            X = [x, x1, x2, x3]
            Y = [y, y1, y2, y3]
        else:
            X = [x]
            Y = [y]

        index = 0
        for i, m in zip(X, Y):
            """ Ensure correct shape """
            i = cv2.resize(i, size)
            if len(i.shape) == 2:  # Ensure single channel
                i = np.expand_dims(i, axis=-1)

            m = cv2.resize(m, size)

            tmp_image_name = f"{name}_{index}.png"
            tmp_mask_name = f"{name}_{index}.png"

            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "mask", tmp_mask_name)

            cv2.imwrite(image_path, i.squeeze(-1))  # Save without errors
            cv2.imwrite(mask_path, m)

            index += 1

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)

    """ Load the data """
    data_path = "/content/drive/MyDrive/unet-oct-dataset-files/"
    (train_x, train_y), (test_x, test_y) = load_data(data_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")

    """ Create directories to save the augmented data """
    create_dir("new_data/train/image/")
    create_dir("new_data/train/mask/")
    create_dir("new_data/test/image/")
    create_dir("new_data/test/mask/")

    """ Data augmentation """
    augment_data(train_x, train_y, "new_data/train/", augment=True)
    augment_data(test_x, test_y, "new_data/test/", augment=False)
    