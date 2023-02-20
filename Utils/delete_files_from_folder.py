import os

path = f'../Dataset/test_augmented_small/img'
images = os.listdir(path)

for image in images:
    os.remove(os.path.join(path, image))
    print(f'{os.path.join(path, image)} removed...')
