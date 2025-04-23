import os
import json
import zipfile
import numpy as np
import cv2
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

def extract_zip(zip_path, extract_to):
    """Extract ZIP files"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def load_annotations(json_path):
    """Load fixation data from JSON"""
    with open(json_path) as f:
        return json.load(f)

def create_heatmap(fixations, img_size=(480, 640), sigma=32):
    """Convert fixations to Gaussian heatmap"""
    heatmap = np.zeros(img_size[::-1], dtype=np.float32)  # (height, width)
    for fixation in fixations:
        x, y = int(fixation['x']), int(fixation['y'])
        if 0 <= x < img_size[0] and 0 <= y < img_size[1]:
            heatmap[y, x] = 1  # Mark fixation points
    return gaussian_filter(heatmap, sigma=sigma)  # Apply Gaussian blur

def process_salicon(data_dir, output_dir):
    """Main processing function"""
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'images/train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'heatmaps/train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images/val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'heatmaps/val'), exist_ok=True)

    # Process train/val splits
    for split in ['train', 'val']:
        print(f"Processing {split} data...")
        
        # Extract images
        extract_zip(os.path.join(data_dir, f'{split}.zip'), 
                   os.path.join(output_dir, 'images', split))
        
        # Load annotations
        annotations = load_annotations(
            os.path.join(data_dir, f'fixations_{split}2014.json'))
        
        # Generate heatmaps
        for img_data in tqdm(annotations['images']):
            img_name = img_data['file_name']
            fixations = [f for f in annotations['annotations'] 
                        if f['image_id'] == img_data['id']]
            
            # Original image path
            img_path = os.path.join(output_dir, 'images', split, img_name)
            
            # Create and save heatmap
            heatmap = create_heatmap(fixations)
            heatmap_path = os.path.join(
                output_dir, 'heatmaps', split, 
                img_name.replace('.jpg', '.png'))
            
            # Normalize and save
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
            cv2.imwrite(heatmap_path, heatmap)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to directory containing SALICON ZIPs/JSONs')
    parser.add_argument('--output_dir', type=str, default='salicon_processed',
                       help='Output directory for processed data')
    args = parser.parse_args()
    
    process_salicon(args.data_dir, args.output_dir)