import os
from skimage import io

dataset_dir = f'../Dataset/training/img'
image_names = os.listdir(dataset_dir)

video_dict = {}

for image_name in image_names:
    img = io.imread(os.path.join(dataset_dir,image_name))
    image_name = image_name.split('.')[0].split('yp')[1]
    if int(image_name) not in video_dict:
        video_dict[int(image_name)] = img.shape

print(video_dict)