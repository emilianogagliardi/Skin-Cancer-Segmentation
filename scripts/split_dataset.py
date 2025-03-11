import os
import shutil
from sklearn.model_selection import train_test_split
import random
import numpy as np

def set_random_seeds(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)

set_random_seeds()

def split_dataset(input_dir, output_dir, train_size=0.6, val_size=0.2, test_size=0.2):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create train, validation, and test directories with images and masks subdirectories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'validation')
    test_dir = os.path.join(output_dir, 'test')
    
    for split_dir in [train_dir, val_dir, test_dir]:
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
        
        # Create images and masks subdirectories in each split
        images_dir = os.path.join(split_dir, 'images')
        masks_dir = os.path.join(split_dir, 'masks')
        
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        if not os.path.exists(masks_dir):
            os.makedirs(masks_dir)
    
    # Get the list of image files
    images_dir = os.path.join(input_dir, 'images')
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg') and not f.startswith('.')]
    
    # Split the dataset
    train_files, temp_files = train_test_split(image_files, train_size=train_size)
    val_files, test_files = train_test_split(temp_files, test_size=test_size/(test_size + val_size))
    
    # Copy images and corresponding masks to their respective directories
    for file_list, target_dir in [(train_files, train_dir), (val_files, val_dir), (test_files, test_dir)]:
        for image_file in file_list:
            # Copy image
            image_src = os.path.join(input_dir, 'images', image_file)
            image_dst = os.path.join(target_dir, 'images', image_file)
            shutil.copy(image_src, image_dst)
            
            # Copy corresponding mask
            # Extract the image ID from the filename (assuming format like ISIC_XXXXXXX.jpg)
            image_id = os.path.splitext(image_file)[0]
            mask_file = f"{image_id}_segmentation.png"
            
            mask_src = os.path.join(input_dir, 'masks', mask_file)
            mask_dst = os.path.join(target_dir, 'masks', mask_file)
            
            if os.path.exists(mask_src):
                shutil.copy(mask_src, mask_dst)
            else:
                print(f"Warning: Mask file {mask_file} not found for image {image_file}")

input_dir = 'data/full_data'
output_dir = 'data/full_data_split'
split_dataset(input_dir, output_dir)
