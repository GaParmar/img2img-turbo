import argparse
from glob import glob
import os
from PIL import Image

def overlay_silhouette_images(image1, image2):
    assert image1.size == image2.size, "Images must be the same size"
    size = image1.size

    # Create first silhouette in blue
    first_silhouette = Image.new('RGB', size, (255, 255, 255))
    blue = Image.new('RGB', size, (0, 0, 255))
    first_silhouette.paste(blue, (0, 0), mask=image1)

    # Create second silhouette in yellow
    second_silhouette = Image.new('RGB', size, (255, 255, 255))
    yellow = Image.new('RGB', size, (255, 255, 0))
    second_silhouette.paste(yellow, (0, 0), mask=image2)

    composite = Image.blend(first_silhouette, second_silhouette, alpha=0.5)
    return composite

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', type=str, default=None, help='path to the test images dataset folder')
    parser.add_argument('--inference_folder', type=str, default=None, help='path to the inference images folder')
    args = parser.parse_args()
    print(args.dataset_folder)

    checkpoint = '751'

    l_images_src_test = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        l_images_src_test.extend(glob(os.path.join(args.dataset_folder, ext)))
    l_images_src_test = sorted(l_images_src_test)

    l_images_results_test = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        l_images_results_test.extend(glob(os.path.join(args.inference_folder, f"fid-{checkpoint}", "samples_a2b", ext)))
    l_images_results_test = sorted(l_images_results_test, key=lambda x: int(os.path.basename(x).split('.')[0]))

    print(len(l_images_src_test), len(l_images_results_test))

    assert len(l_images_src_test) == len(l_images_results_test)

    for i, (src, res) in enumerate(zip(l_images_src_test, l_images_results_test)):
        src_img = Image.open(src).convert('L')
        res_img = Image.open(res).resize((src_img.width, src_img.height), Image.LANCZOS).convert('L')
        src_img.show()
        res_img.show()
        composite = overlay_silhouette_images(src_img, res_img)
        composite.show()
        if i >= 5:
            break
