import os

dataset_dir = f'../Dataset/dataset_all/mask'
image_names = os.listdir(dataset_dir)

for image_name in image_names:
    image_name_new = image_name.replace('_mask', '')
    os.rename(f'{dataset_dir}/{image_name}', f'{dataset_dir}/{image_name_new}')