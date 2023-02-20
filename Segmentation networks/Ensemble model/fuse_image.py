import os

from skimage import io, transform, exposure
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

image_path = f'../../Dataset/test/img'
gt_path = f'../../Dataset/test/mask'

pranet_path = 'C://Users//snorb//PycharmProjects//PraNet-master//PraNet-master//results//PraNet//{test}'
pvt_path = 'C://Users//snorb//PycharmProjects//Polyp-PVT-main//result_map//PolypPVT//test'
colon_path = 'C://Users//snorb//PycharmProjects//ColonSegNet-main//ColonSegNet-main//results//mask'
ens_result_path = f'results/mask'

def map_values_to_rgb(pred):
    value_dict = {1: [0,255,0]}
    res = np.zeros((pred.shape[0], pred.shape[1], 3))
    for k,v in value_dict.items():
        res[pred == k] = v
    return res

image_names = [image_name for image_name in os.listdir(image_path)]
for i, image_name in enumerate(image_names):
    rgb_image = io.imread(os.path.join(image_path,image_name))

    gt_image = io.imread(os.path.join(gt_path,image_name))
    gt_image = exposure.rescale_intensity(gt_image, out_range=(0,1))

    pranet_image = io.imread(os.path.join(pranet_path, image_name), as_gray = True)
    pranet_image = exposure.rescale_intensity(pranet_image, out_range=(0,1))
    pvt_image = io.imread(os.path.join(pvt_path, f'image_{i}.png'), as_gray = True)
    pvt_image = exposure.rescale_intensity(pvt_image, out_range=(0,1))
    colon_image = io.imread(os.path.join(colon_path, f'{image_name}.png'), as_gray = True)
    colon_image = transform.resize(colon_image, (rgb_image.shape[0], rgb_image.shape[1]))
    ens_image = io.imread(os.path.join(ens_result_path, f'image_{i}.png'), as_gray = True)
    ens_image = transform.resize(ens_image, (rgb_image.shape[0], rgb_image.shape[1]))

    fig, ax = plt.subplots(2,3)
    ax[0,0].imshow(rgb_image)
    ax[0,1].set_title('Ground Truth')
    ax[0,1].imshow(np.array(Image.blend(Image.fromarray(rgb_image.astype(np.uint8)), Image.fromarray(map_values_to_rgb(gt_image).astype(np.uint8)), 0.35)))
    ax[0,2].set_title('Pranet')
    ax[0,2].imshow(np.array(Image.blend(Image.fromarray(rgb_image.astype(np.uint8)), Image.fromarray(map_values_to_rgb(pranet_image).astype(np.uint8)), 0.35)))
    ax[1,0].set_title('PolypPVT')
    ax[1,0].imshow(np.array(Image.blend(Image.fromarray(rgb_image.astype(np.uint8)), Image.fromarray(map_values_to_rgb(pvt_image).astype(np.uint8)), 0.35)))
    ax[1,1].set_title('ColonSegNet')
    ax[1,1].imshow(np.array(Image.blend(Image.fromarray(rgb_image.astype(np.uint8)), Image.fromarray(map_values_to_rgb(colon_image).astype(np.uint8)), 0.35)))
    ax[1,2].set_title('Proposed')
    ax[1,2].imshow(np.array(Image.blend(Image.fromarray(rgb_image.astype(np.uint8)), Image.fromarray(map_values_to_rgb(ens_image).astype(np.uint8)), 0.35)))
    plt.savefig(f'results/figures/{i}.png', dpi = 300)