import os
import shutil

test_videos = [3,4]
validation_videos = [1,9]

# img_dir = f'dataset_all_/img'
# mask_dir = f'dataset_all/mask'

# augmented
img_dir = f'dataset_all_augmented/img'
mask_dir = f'dataset_all_augmented/mask'

image_names = os.listdir(img_dir)

for image_name in image_names:
    image_num = int(image_name.split('.')[0].split('yp')[1])
    # if image_num not in test_videos and image_num not in validation_videos:
    #     shutil.copyfile(f'{img_dir}/{image_name}', f'training/img/{image_name}')
    #     shutil.copyfile(f'{mask_dir}/{image_name}', f'training/mask/{image_name}')
    # elif image_num not in test_videos:
    #     shutil.copyfile(f'{img_dir}/{image_name}', f'validation/img/{image_name}')
    #     shutil.copyfile(f'{mask_dir}/{image_name}', f'validation/mask/{image_name}')

    # augmented
    if image_num not in test_videos and image_num not in validation_videos:
        shutil.copyfile(f'{img_dir}/{image_name}', f'training_augmented/img/{image_name}')
        shutil.copyfile(f'{mask_dir}/{image_name}', f'training_augmented/mask/{image_name}')
    elif image_num not in test_videos:
        shutil.copyfile(f'{img_dir}/{image_name}', f'validation_augmented/img/{image_name}')
        shutil.copyfile(f'{mask_dir}/{image_name}', f'validation_augmented/mask/{image_name}')
    else:
        shutil.copyfile(f'{img_dir}/{image_name}', f'test_augmented/img/{image_name}')
        shutil.copyfile(f'{mask_dir}/{image_name}', f'test_augmented/mask/{image_name}')
