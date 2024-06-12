#!/usr/bin/env python3

" makes crops to populate dataset "

import os
import argparse
from PIL import Image
from tqdm import tqdm


def is_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except:
        return False


def create_directories(output_dir, category):
    os.makedirs(os.path.join(output_dir, category, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, category, 'masks'), exist_ok=True)


def process_images(input_dir, output_dir, category, crop_size, overlap, img_extension, mask_extension):
    image_list_path = os.path.join(input_dir, f'{category}.txt')
    create_directories(output_dir, category)

    with open(image_list_path, 'r') as file_list:
        for line in tqdm(file_list, desc=f"Processing {category} data"):
            image_name = line.strip()
            image_path = os.path.join(input_dir, category, 'images', f'{image_name}{img_extension}')
            mask_path = os.path.join(input_dir, category, 'masks', f'{image_name}{mask_extension}')

            if not check_image_validity(image_path):
                print(f"Invalid image: {image_path}")
                continue

            image = Image.open(image_path)
            mask = Image.open(mask_path)
            image_width, image_height = image.size

            x_max = (image_width - crop_size) // overlap + 1
            y_max = (image_height - crop_size) // overlap + 1

            for x in range(x_max):
                for y in range(y_max):
                    left = x * overlap
                    upper = y * overlap
                    right = left + crop_size
                    lower = upper + crop_size

                    image_crop = image.crop((left, upper, right, lower))
                    mask_crop = mask.crop((left, upper, right, lower))

                    crop_name = f'{image_name}_{left}_{upper}.png'
                    image_crop.save(os.path.join(output_dir, category, 'images', crop_name))
                    mask_crop.save(os.path.join(output_dir, category, 'masks', crop_name))


def main():
    parser = argparse.ArgumentParser(description="Generate crops from images and masks for machine learning datasets.")
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing the dataset.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory to save the cropped images and masks.')
    parser.add_argument('--crop_size', type=int, required=True, help='Size of the square crop.')
    parser.add_argument('--overlap', type=int, required=True, help='Overlap size between crops.')
    parser.add_argument('--img_extension', type=str, required=True, help='File extension of images (e.g., .jpg, .png).')
    parser.add_argument('--mask_extension', type=str, required=True, help='File extension of masks (e.g., .png).')

    args = parser.parse_args()

    process_images(args.input_dir, args.output_dir, 'train', args.crop_size, args.overlap, args.img_extension, args.mask_extension)
    process_images(args.input_dir, args.output_dir, 'val', args.crop_size, args.crop_size, args.img_extension, args.mask_extension)


if __name__ == "__main__":
    main()
