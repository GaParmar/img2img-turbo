import os
import argparse
from PIL import Image
from training_utils import build_transform

def convert_images_to_binary(input_dir):
    """
    Converts all PNG images in the specified directory to binary format using grayscale thresholding.
    
    Args:
        input_dir (str): Directory containing PNG images to convert
    """
    # Build the grayscale to binary transform
    transform = build_transform("grayscale_to_binary")
    
    # Process each PNG file in the directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.png'):
            try:
                # Open image
                img_path = os.path.join(input_dir, filename)
                img = Image.open(img_path)
                
                # Apply transformation
                binary_img = transform(img)
                
                # Save the binary image (overwriting the original)
                binary_img.save(img_path)
                
                print(f"Converted {filename} to binary")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Convert PNG images to binary format')
    parser.add_argument('input_dir', type=str, help='Directory containing PNG images to convert')
    args = parser.parse_args()
    
    # Convert images
    convert_images_to_binary(args.input_dir)

if __name__ == '__main__':
    main()
