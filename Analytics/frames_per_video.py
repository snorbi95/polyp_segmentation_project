import os

dataset_dir = f'../Dataset/training/img'
image_names = os.listdir(dataset_dir)

video_dict = {}

for image_name in image_names:
    image_name = image_name.split('.')[0].split('yp')[1]
    if int(image_name) not in video_dict:
        video_dict[int(image_name)] = 1
    else:
        video_dict[int(image_name)] += 1

print(video_dict)