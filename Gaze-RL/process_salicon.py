import os
import json
import zipfile
import numpy as np
import cv2
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from multiprocessing import Pool, cpu_count
from collections import defaultdict
import shutil

def extract_zip(zip_path, extract_to):
    """Smart extraction that handles nested COCO folders"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if file.endswith('.jpg'):
                # Extract maintaining original paths
                zip_ref.extract(file, extract_to)
                
                # Flatten structure (optional)
                src = os.path.join(extract_to, file)
                dst = os.path.join(extract_to, os.path.basename(file))
                shutil.move(src, dst)
    
    # Clean empty subfolders
    for root, dirs, _ in os.walk(extract_to, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)

def load_annotations(json_path):
    """Load and validate annotations with fixations array"""
    with open(json_path) as f:
        data = json.load(f)
    
    # Validate structure
    required_keys = {'info', 'images', 'annotations', 'licenses'}
    if not required_keys.issubset(data.keys()):
        raise ValueError("Invalid annotation format")
    
    # Check annotation structure
    for ann in data['annotations']:
        if 'fixations' not in ann or not isinstance(ann['fixations'], list):
            raise ValueError("Annotations must contain 'fixations' arrays")
    
    return data

def process_single_image(args):
    """Parallel processing for one image with fixations"""
    img_data, annotations, output_dir, split = args
    img_name = img_data['file_name']
    img_path = os.path.join(output_dir, 'images', split, img_name)
    
    # Skip if image missing
    if not os.path.exists(img_path):
        return None
    
    # Initialize heatmap
    heatmap = np.zeros((img_data['height'], img_data['width']), dtype=np.float32)
    
    # Accumulate fixations for this image
    for ann in annotations:
        if ann['image_id'] == img_data['id']:
            for row, col in ann['fixations']:
                x, y = int(col), int(row)  # Note: SALICON uses (row,col) = (y,x)
                if 0 <= x < img_data['width'] and 0 <= y < img_data['height']:
                    heatmap[y, x] += 1
    
    # Apply Gaussian smoothing
    heatmap = gaussian_filter(heatmap, sigma=32)
    
    # Normalize and save
    if heatmap.max() > 0:
        heatmap = (255 * heatmap / heatmap.max()).astype(np.uint8)
    
    heatmap_path = os.path.join(
        output_dir, 'heatmaps', split, 
        img_name.replace('.jpg', '.png'))
    cv2.imwrite(heatmap_path, heatmap)
    
    return img_name

def process_split(data_dir, output_dir, split):
    """Process a dataset split with fixations"""
    print(f"\nProcessing {split} data...")
    
    # Path setup
    zip_path = os.path.join(data_dir, f'{split}.zip')
    json_path = os.path.join(data_dir, f'fixations_{split}2014.json')
    
    # Extract images
    os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
    extract_zip(zip_path, os.path.join(output_dir, 'images', split))
    
    # Load annotations
    data = load_annotations(json_path)
    annotations = data['annotations']
    
    # Prepare parallel tasks
    os.makedirs(os.path.join(output_dir, 'heatmaps', split), exist_ok=True)
    task_args = [
        (img, annotations, output_dir, split)
        for img in data['images']
    ]
    
    # Process with progress
    successful = 0
    # with Pool(processes=max(1, cpu_count()-1)) as pool:
    # for result in tqdm(pool.imap(process_single_image, task_args), 
    #                     total=len(task_args), 
    #                     desc=f"Processing {split}"):
        # if result is not None:
        #     successful += 1
    for args in tqdm(task_args, total=len(task_args), desc=f"Processing {split}"):
        result = process_single_image(args)
        if result is not None:
            successful += 1

    
    print(f"Completed: {successful}/{len(data['images'])} images")

def process_salicon(data_dir, output_dir):
    """Main processing function"""
    # Validate inputs
    required_files = {
        'train': ['train.zip', 'fixations_train2014.json'],
        'val': ['val.zip', 'fixations_val2014.json']
    }
    
    for split, files in required_files.items():
        for file in files:
            if not os.path.exists(os.path.join(data_dir, file)):
                raise FileNotFoundError(f"Missing {file}")
    
    # Process both splits
    for split in ['train', 'val']:
        process_split(data_dir, output_dir, split)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True,
                       help='Directory with SALICON ZIPs/JSONs')
    parser.add_argument('--output_dir', default='salicon_processed',
                       help='Output directory')
    args = parser.parse_args()
    
    try:
        process_salicon(args.data_dir, args.output_dir)
        print("\nProcessing completed successfully!")
    except Exception as e:
        print(f"\nError: {str(e)}")