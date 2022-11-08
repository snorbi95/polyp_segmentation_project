import math

from skimage.measure import regionprops, label
from skimage import io
import os
import matplotlib.pyplot as plt

image_dir = f'../Dataset/dataset_all/img'
mask_dir = f'../Dataset/dataset_all/mask'

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

def image_cropper(image_name, slices_to_capture):
    rgb_image = io.imread(os.path.join(image_dir, image_name))
    mask_image = io.imread(os.path.join(mask_dir, image_name))
    label_image = label(mask_image)

    regions = regionprops(label_image)
    regions.sort(key=lambda x: x.area, reverse=True)
    region = regions[0]

    y0, x0 = region.centroid
    orientation = region.orientation
    x1 = x0 + math.sin(orientation) * 0.5 * region.axis_major_length
    y1 = y0 + math.cos(orientation) * 0.5 * region.axis_major_length
    x2 = x0 - math.sin(orientation) * 0.5 * region.axis_major_length
    y2 = y0 - math.cos(orientation) * 0.5 * region.axis_major_length
    x_interval = abs(x2 - x1) / slices_to_capture
    y_interval = abs(y2 - y1) / slices_to_capture

    if x2 < x1:
        x_interval *= -1
    if y2 < y1:
        y_interval *= -1

    captured_images = 0
    while captured_images < slices_to_capture:
        for i in range(slices_to_capture + 1):
            # ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
            # plt.plot((x1, x2), (y1, y2), '-r', linewidth=2.5)
            x, y = int(x1 + (i * x_interval)), int(y1 + (i * y_interval))
            if x - 192 > 0 and y - 192 > 0 and x + 192 < mask_image.shape[1] and y + 192 < mask_image.shape[0]:
                captured_images += 1
                cropped_mask = mask_image[y - 192:y + 192,x - 192:x + 192]
                cropped_rgb_image = rgb_image[y - 192:y + 192,x - 192:x + 192]
                io.imsave(f'../Dataset/dataset_all_augmented/img/{image_name}_{captured_images}.png', cropped_rgb_image)
                io.imsave(f'../Dataset/dataset_all_augmented/mask/{image_name}_{captured_images}.png', cropped_mask)
                print(f'Saving {captured_images}th image from {image_name}...')
                #plt.imshow(cropped_mask)
                #plt.imshow(mask_image)
                # plt.plot((x - 128, x + 128), (y - 128, y - 128))
                # plt.plot((x - 128, x + 128), (y + 128, y + 128))
                # plt.plot((x + 128, x + 128), (y - 128, y + 128))
                # plt.plot((x - 128, x - 128), (y + 128, y - 128))
                # # plt.plot((x, x + 128), (y, y))
                # # plt.plot((x, x), (y, y - 128))
                # # plt.plot((x, x), (y, y + 128))
                # plt.plot(x, y, 'go')
                # plt.plot()
                # ax.axis((0,600,600,0))
                #plt.show()
        if get_image_quarter(x1, y1, mask_image.shape) == 1:
            x1, y1 = x1 + 15, y1 + 15
        elif get_image_quarter(x1, y1, mask_image.shape) == 2:
            x1, y1 = x1 - 15, y1 + 15
        elif get_image_quarter(x1, y1, mask_image.shape) == 3:
            x1, y1 = x1 + 15, y1 - 15
        else:
            x1, y1 = x1 - 15, y1 - 15

# Test image quartering function
# x = 500
# y = 500
# shape = [600,700]
# print(get_image_quarter(x,y,shape))
# plt.axis((0,600,700,0))
# plt.plot((shape[0] // 2,shape[0] // 2),(0, shape[1]))
# plt.plot((0, shape[0]),(shape[1] // 2,shape[1] // 2))
# plt.plot(x,y,'go')
# plt.show()

video_number_dict = {}

for image_name in image_names:
    image_num = int(image_name.split('.')[0].split('yp')[1])
    if image_num not in video_number_dict:
        video_number_dict[image_num] = 1
    else:
        video_number_dict[image_num] += 1


for k,v in video_number_dict.items():
    image_names_from_video = [image_name for image_name in image_names if int(image_name.split('.')[0].split('yp')[1]) == k]
    if len(image_names_from_video) != 0:
        slices_per_image = 500 // len(image_names_from_video)
        last_slices_plus = 500 % len(image_names_from_video)
        for i, image_name_from_video in enumerate(image_names_from_video):
            if i == len(image_names_from_video) - 1:
                current_number_of_slices = slices_per_image + last_slices_plus
            else:
                current_number_of_slices = slices_per_image
            image_cropper(image_name_from_video, current_number_of_slices)

