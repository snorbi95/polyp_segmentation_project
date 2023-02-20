import math
import random

from skimage.measure import regionprops, label
from skimage import io
import os
import matplotlib.pyplot as plt

image_dir = f'../Dataset/dataset_all_augmented/img'
mask_dir = f'../Dataset/dataset_all_augmented/mask'
image_size = 192

image_names = os.listdir(image_dir)

def get_image_quarter(x,y,shape):
    if x < shape[0] // 2 and y < shape[1] // 2:
        return 1
    elif x > shape[0] // 2 and y < shape[1] // 2:
        return 2
    elif x < shape[0] // 2 and y > shape[1] // 2:
        return 3
    else:
        return 4

def image_cropper(image_name):
    rgb_image = io.imread(os.path.join(image_dir, image_name))
    mask_image = io.imread(os.path.join(mask_dir, image_name))

    n = 0
    for i in [192, 384]:
        for j in [192, 384]:
            cropped_rgb_image = rgb_image[i - 192:i, j - 192:j, :]
            cropped_mask = mask_image[i - 192:i, j - 192:j]
            io.imsave(f'../Dataset/dataset_all_augmented_small/img/{image_name}_{n}.png', cropped_rgb_image)
            io.imsave(f'../Dataset/dataset_all_augmented_small/mask/{image_name}_{n}.png', cropped_mask)
            print(f'Saving image {image_name}_{n}.png...')
            n += 1

for image_name in image_names:
    image_cropper(image_name)

