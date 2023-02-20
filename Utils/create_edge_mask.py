import os
from skimage import io, feature, morphology
import matplotlib.pyplot as plt
import numpy as np

mask_dir = f'../Dataset/dataset_all/mask'
mask_image_names = os.listdir(mask_dir)

for image_name in mask_image_names:
    image = io.imread(os.path.join(mask_dir, image_name))
    edge_image = feature.canny(image)
    edge_image = (morphology.dilation(edge_image, selem = morphology.square(6)) * 255).astype(np.uint8)
    # plt.imshow(edge_image)
    # plt.show()
    io.imsave(f'../Dataset/dataset_all/mask_edge/{image_name}', edge_image)
    print(f'Saving image {image_name}...')