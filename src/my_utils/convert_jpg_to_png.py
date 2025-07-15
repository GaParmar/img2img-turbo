#!/usr/bin/env python3
"""
Utility script to convert JPG/JPEG images to PNG format and remove the originals.
Preserves directory structure while converting and cleaning up.

Usage:
    python -m my_utils.convert_jpg_to_png /path/to/your/directory
"""

import os
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def convert_jpg_to_png(directory):
    """
    Convert all JPG/JPEG files in the directory (and subdirectories) to PNG format.
    
    Args:
        directory (str): Path to the directory containing images
    """
    # Get all JPG/JPEG files recursively
    image_extensions = {'.jpg', '.jpeg', '.JPG', '.JPEG'}
    jpg_files = []
    for ext in image_extensions:
        jpg_files.extend(Path(directory).rglob(f'*{ext}'))
    
    if not jpg_files:
        print(f"No JPG/JPEG files found in {directory}")
        return
    
    print(f"Found {len(jpg_files)} JPG/JPEG files to convert...")
    
    # Convert each file
    converted = 0
    skipped = 0
    errors = 0
    
    for jpg_path in tqdm(jpg_files, desc="Converting images"):
        try:
            # Create output path with .png extension
            png_path = jpg_path.with_suffix('.png')
            
            # Skip if PNG already exists
            if png_path.exists():
                print(f"Skipping (PNG exists): {jpg_path}")
                skipped += 1
                continue
                
            # Open and convert the image
            with Image.open(jpg_path) as img:
                # Convert to RGB if necessary (for CMYK JPGs)
                if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                    img = img.convert('RGBA')
                else:
                    img = img.convert('RGB')
                
                # Save as PNG
                img.save(png_path, 'PNG', quality=100)
                
                # Remove original JPG after successful conversion
                try:
                    os.remove(str(jpg_path))  # Convert Path to string for os.remove
                    converted += 1
                except Exception as e:
                    print(f"Warning: Could not remove {jpg_path}: {str(e)}")
                    converted += 1  # Still count as converted since PNG was created
                
        except Exception as e:
            print(f"Error converting {jpg_path}: {str(e)}")
            errors += 1
            continue
    
    print(f"\nConversion complete!")
    print(f"Converted and removed: {converted}")
    print(f"Skipped (PNG exists): {skipped}")
    if errors > 0:
        print(f"Errors: {errors} (original files preserved for these)")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_jpg_to_png.py /path/to/your/directory")
        sys.exit(1)
    
    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"Error: Directory not found: {directory}")
        sys.exit(1)
    
    convert_jpg_to_png(directory)
