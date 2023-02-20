import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from Utils.metrics import jaccard_score, jaccard_score_true_class, dice_coef, dice_coef_true_class
from Utils.losses import weighted_jaccard_loss, jaccard_loss
import os
from pathlib import Path
from Utils.dataset import Dataset, Dataloder
from Utils.visuals import visualize_edges, visualize, denormalize
from Utils.augmentation import get_preprocessing
import segmentation_models as sm
import numpy as np

img_size = (256, 256)
CLASSES = ['background', 'polyp']
metrics = [jaccard_score_true_class, jaccard_score, dice_coef]
optim = keras.optimizers.Adam(0.001)
preprocess_input_efficientnet = sm.get_preprocessing('efficientnetb3')
preprocess_input_seresnet = sm.get_preprocessing('seresnet18')
preprocess_input_mobilenet = sm.get_preprocessing('mobilenetv2')
num_classes = 2

DATA_DIR = f'{Path(__file__).parent.parent.parent}/Dataset'

x_test_dir = os.path.join(DATA_DIR, 'test/img')
y_test_dir = os.path.join(DATA_DIR, 'test/mask')
test_len = len(os.listdir(x_test_dir))

test_dataset_efficientnet = Dataset(
    x_test_dir,
    y_test_dir,
    img_size = img_size,
    classes=CLASSES,
    preprocessing=get_preprocessing(preprocess_input_efficientnet),
)
test_dataloader_efficientnet = Dataloder(test_dataset_efficientnet, batch_size=1, shuffle=False, train_len=test_len)

test_dataset_mobilenet = Dataset(
    x_test_dir,
    y_test_dir,
    img_size = img_size,
    classes=CLASSES,
    preprocessing=get_preprocessing(preprocess_input_mobilenet),
)
test_dataloader_mobilenet = Dataloder(test_dataset_mobilenet, batch_size=1, shuffle=False, train_len=test_len)

test_dataset_blank = Dataset(
    x_test_dir,
    y_test_dir,
    img_size = img_size,
    classes=CLASSES,
    #preprocessing=get_preprocessing(preprocess_input_efficientnet),
)
test_dataloader_blank = Dataloder(test_dataset_blank, batch_size=1, shuffle=False, train_len=test_len)

test_dataset_seresnet = Dataset(
    x_test_dir,
    y_test_dir,
    img_size = img_size,
    classes=CLASSES,
    preprocessing=get_preprocessing(preprocess_input_seresnet),
)
test_dataloader_seresnet = Dataloder(test_dataset_seresnet, batch_size=1, shuffle=False, train_len=test_len)

test_dataset_pspnet = Dataset(
    x_test_dir,
    y_test_dir,
    img_size = (240,240),
    classes=CLASSES,
    preprocessing=get_preprocessing(preprocess_input_efficientnet),
)
test_dataloader_pspnet = Dataloder(test_dataset_pspnet, batch_size=1, shuffle=False, train_len=test_len)

unet_model_efficientnet = keras.models.load_model(f'../U-Net/models/unet_1_efficientnetb3_jaccard_loss_256_size_cropped_25_epoch_set_#5.h5',
                                                  custom_objects={'jaccard_loss': jaccard_loss, 'jaccard_score_true_class': jaccard_score_true_class,
                                                  'jaccard_score': jaccard_score,'dice_coef': dice_coef})

unet_model_seresnet = keras.models.load_model(f'../U-Net/models/unet_1_seresnet18_jaccard_loss_256_size_cropped_25_epoch_set_#4.h5',
                                                  custom_objects={'jaccard_loss': jaccard_loss, 'jaccard_score_true_class': jaccard_score_true_class,
                                                  'jaccard_score': jaccard_score,'dice_coef': dice_coef})

deeplab_model = keras.models.load_model(f'../Deeplab/models/deeplab_1_ssim_loss_256_size_efficientnetb3_cropped_15_epoch_set_#5.h5',
                                                    custom_objects={'jaccard_loss': jaccard_loss, 'jaccard_score_true_class': jaccard_score_true_class,
                                                  'jaccard_score': jaccard_score,'dice_coef': dice_coef})

pspnet_model = keras.models.load_model(f'../PSPNet/models/pspnet_1_jaccard_loss_240_size_cropped_20_epoch_set_#5.h5',
                                                    custom_objects={'jaccard_loss': jaccard_loss, 'jaccard_score_true_class': jaccard_score_true_class,
                                                  'jaccard_score': jaccard_score,'dice_coef': dice_coef})


def weighted_prediction(p1, p2, weights = [0.7717,0.743]):
    pixel_prediction = np.zeros((img_size[0], img_size[1]))
    values_p1 = np.max(p1, axis=-1)
    indices_p1 = np.argmax(p1, axis=-1)

    values_p2 = np.max(p2, axis=-1)
    indices_p2 = np.argmax(p2, axis=-1)

    values = np.zeros((1,256,256,3))
    values[:,:,:,0] = values_p1
    values[:, :, :, 1] = values_p2

    indices = np.zeros((1,256,256,3))
    indices[:,:,:,0] = indices_p1
    indices[:, :, :, 1] = indices_p2

    for i in range(img_size[0]):
        for j in range(img_size[1]):
            model_to_choose = np.argmax([weights[0] * values[0,i,j,0],weights[1] * values[0,i,j,1]])
            pixel_prediction[i,j] = indices[0,i,j,model_to_choose]

    return pixel_prediction

def check_threes(x,y,image):
    sum = 0
    for i in range(x - 1,x + 2):
        for j in range(y - 1, y + 2):
            try:
                if image[i,j] >= 2.5:
                    sum += 1
            except Exception as e:
                continue
    return sum

def reach_edge(x,y, image):
    distance_map = np.indices((image.shape[0],image.shape[1]))
    d_map = np.zeros((image.shape[0], image.shape[0], 2))
    d_map[:, :, 0] = distance_map[0]
    d_map[:, :, 1] = distance_map[1]

    dist = np.sqrt(((d_map[:,:,0] - x) ** 2) + ((d_map[:,:,1] - y) ** 2))
    dist[image == image[x,y]] = image.shape[0] + image.shape[1]
    # print(x,y,np.where(dist == np.min(dist))[0][0], np.where(dist == np.min(dist))[1][0], np.min(dist))
    dist[(image > 1.5) & (image < 2.5)] = 512
    # plt.imshow(dist, cmap = 'gray')
    # plt.plot(x,y, color = 'r', markersize = 6)
    # plt.show()

    # for i in range(distance_map.shape[0]):
    #     for j in range(distance_map.shape[1]):
    #         if image[i,j] != image[x,y]:
    #             distance_map[i,j] = np.linalg.norm(np.array((i,j)) - np.array((x,y)))
    #         else:
    #             distance_map[i, j] = np.sqrt(image.shape[0] ** 2 + image.shape[1] ** 2)

    #print(np.where(distance_map == np.min(distance_map)))
    i,j = np.where(dist == np.min(dist))[0][0], np.where(dist == np.min(dist))[1][0]
    # print(i,j)
    if image[i,j] < 2:
        return 1, i,j,np.min(dist), dist
    return 3, i,j,np.min(dist), dist
    # delta_x, delta_y = 0, 0
    # threes, ones = 0, 0
    # while True:
    #     if x + delta_x < image.shape[0] and y + delta_y < image.shape[1]:
    #         if image[x + delta_x, y + delta_y] > 2.5:
    #             threes += 1
    #         if image[x + delta_x, y + delta_y] < 1.5:
    #             ones += 1
    #     if x + delta_x < image.shape[0] and y - delta_y >= 0:
    #         if image[x + delta_x, y - delta_y] > 2.5:
    #             threes += 1
    #         if image[x + delta_x, y - delta_y] < 1.5:
    #             ones += 1
    #     if x - delta_x >= 0 and y + delta_y < image.shape[1]:
    #         if image[x - delta_x, y + delta_y] > 2.5:
    #             threes += 1
    #         if image[x - delta_x, y + delta_y] < 1.5:
    #             ones += 1
    #     if x - delta_x >= 0 and y - delta_y >= 0:
    #         if image[x - delta_x, y - delta_y] > 2.5:
    #             threes += 1
    #         if image[x - delta_x, y - delta_y] < 1.5:
    #             ones += 1
    #     if threes != 0 or ones != 0:
    #         if threes > ones:
    #             return 3
    #         return 1
    #     #print(threes, ones)
    #     delta_x, delta_y = delta_x + 1, delta_y + 1



def median_prediction(p1, p2, p3, image_num = 0):
    from skimage import feature, morphology, transform
    from scipy.ndimage import binary_fill_holes

    p2 = np.argmax(p2, axis=-1)
    p3 = np.argmax(p3, axis = -1)
    p2 = p2[0,:,:]
    p3 = p3[0,:,:]
    # plt.imshow(p3)
    # plt.show()
    p3 = transform.resize(p3, output_shape=img_size, preserve_range=True)
    sum_prediction = p1 + p2 + p3
    # edge_p1 = feature.canny(p1).astype(np.uint8)
    # edge_p2 = feature.canny(p2).astype(np.uint8)
    # edge_p3 = feature.canny(p3).astype(np.uint8)
    # sum_edges = edge_p1 + edge_p2 + edge_p3
    # ones = np.zeros_like(sum_prediction)
    # ones[sum_prediction == 1] = 1

    twos = np.zeros_like(sum_prediction)
    twos[(sum_prediction > 1.5) & (sum_prediction < 2.5)] = 1
    from skimage.measure import label, regionprops
    label_image = label(twos)

    # plt.imshow(sum_prediction, cmap = 'gray')
    # plt.show()

    regions = regionprops(label_image)
    # for region in regions:
    #     sum_of_threes = 0
    #     for x,y in region.coords:
    #         sum_of_threes += check_threes(x,y, sum_prediction)
    #     print(sum_of_threes, region.perimeter)
    #     sub_image = sum_prediction[region.bbox[0]:region.bbox[2],region.bbox[1]:region.bbox[3]]
    #     if sum_of_threes > region.perimeter:
    #         sub_image[(sub_image > 1.5) & (sub_image < 2.5)] = 3
    #         sum_prediction[region.bbox[0]:region.bbox[2], region.bbox[1]:region.bbox[3]] = sub_image
        # plt.imshow(sum_prediction[region.bbox[0]:region.bbox[2],region.bbox[1]:region.bbox[3]])
        # plt.show()
    original_pred_image = np.copy(sum_prediction)
    for i, region in enumerate(regions):
        print(i)
        showed = False
        for x,y in region.coords:
            sum_prediction[x,y],x_d,y_d, d, d_map = reach_edge(x,y, original_pred_image)
            if d > 5 and x > sum_prediction.shape[0] * 0.75 and y > sum_prediction.shape[1] // 2 and image_num == 3 and not showed:
                d_map [d_map == 512] = 0
                d_map = d_map * 5
                fig,ax = plt.subplots(1,3)
                ax[0].imshow(original_pred_image, cmap = 'gray')
                ax[1].imshow(d_map, cmap = 'gray')
                ax[1].plot(y,x,'go')
                ax[1].plot(y_d, x_d,'ro')
                ax[1].plot((y,y_d),(x,x_d))
                ax[2].imshow(original_pred_image, cmap = 'gray')
                ax[2].plot(y,x,'go')
                ax[2].plot(y_d, x_d,'ro')
                ax[2].plot((y,y_d),(x,x_d))
                plt.show()
                showed = True
    # fig, ax = plt.subplots(1,2)
    # ax[0].imshow(original_pred_image)
    # ax[1].imshow(sum_prediction)
    # plt.show()

    threes = np.zeros_like(sum_prediction)
    threes[sum_prediction == 3] = 1

    # fig, ax = plt.subplots(2,4)
    # ax[0,0].imshow(p1)
    # ax[0,1].imshow(p2)
    # ax[0,2].imshow(p3)
    # ax[0,3].imshow(sum_prediction)
    # ax[1,0].imshow(ones)
    # ax[1,1].imshow(twos)
    # ax[1,2].imshow(threes)
    # ax[1,3].imshow(sum_prediction)
    # plt.show()
    sum_prediction[sum_prediction < 1.5] = 0
    sum_prediction[sum_prediction != 0] = 1

    plt.imshow(sum_prediction, cmap = 'gray')
    plt.show()
    #
    # fig, ax = plt.subplots(1,2)
    # ax[0].imshow(sum_prediction)
    # ax[1].imshow(binary_fill_holes(sum_prediction))
    # plt.show()

    #return morphology.closing(sum_prediction, selem = morphology.disk(5))
    return binary_fill_holes(sum_prediction)

import keras.backend as K
import segmentation_models as sm
ids = len(test_dataloader_efficientnet)
jaccard_score_true_class_unet_efficientnet = 0
jaccard_score_unet_efficientnet = 0
dice_score_true_class_unet_efficicentnet = 0
dice_score_unet_efficicentnet = 0
precision_unet_efficientnet = 0
recall_unet_efficientnet = 0
f2_score_unet_efficientnet = 0

jaccard_score_true_class_unet_seresnet = 0
jaccard_score_unet_seresnet = 0
dice_score_true_class_unet_seresnet = 0
dice_score_unet_seresnet = 0
precision_unet_seresnet = 0
recall_unet_seresnet = 0
f2_score_unet_seresnet = 0

jaccard_score_true_class_deeplab = 0
jaccard_score_deeplab = 0
dice_score_true_class_deeplab = 0
dice_score_deeplab = 0
precision_deeplab = 0
recall_deeplab = 0
f2_score_deeplab = 0

jaccard_score_true_class_pspnet = 0
jaccard_score_pspnet = 0
dice_score_true_class_pspnet = 0
dice_score_pspnet = 0
precision_pspnet = 0
recall_pspnet = 0
f2_score_pspnet = 0

jaccard_score_true_class_unet_weighted = 0
jaccard_score_unet_weighted = 0
dice_score_true_class_unet_weighted = 0
dice_score_unet_weighted = 0
precision_unet_weighted = 0
recall_unet_weighted = 0
f2_score_unet_weighted = 0

jaccard_score_true_class_median_prediction = 0
jaccard_score_median_prediction = 0
dice_score_true_class_median_prediction = 0
dice_score_median_prediction = 0
precision_median_prediction = 0
recall_median_prediction = 0
f2_score_median_prediction = 0

for i in range(ids):
    print(f'Prediction on image #{i}...')
    num = i#np.random.randint(0,ids)
    image_efficientnet, y = test_dataset_efficientnet[num]
    image_efficientnet = np.expand_dims(image_efficientnet, axis=0)

    image_mobilenet, y = test_dataset_mobilenet[num]
    image_mobilenet = np.expand_dims(image_mobilenet, axis=0)

    image, y = test_dataset_blank[num]
    image = np.expand_dims(image, axis = 0)

    image_pspnet, y_psp = test_dataset_pspnet[num]
    image_pspnet = np.expand_dims(image_pspnet, axis = 0)

    indices = np.zeros((img_size[0], img_size[1], 3))
    values = np.zeros((img_size[0], img_size[1], 3))


    image_seresnet, y = test_dataset_seresnet[num]
    image_seresnet = np.expand_dims(image_seresnet, axis=0)

    p_unet_efficientnet = unet_model_efficientnet.predict(image_efficientnet)
    p_unet_seresnet = unet_model_seresnet.predict(image_seresnet)
    p_deeplab = deeplab_model.predict(image)
    p_pspnet = pspnet_model.predict(image_pspnet)

    pixel_prediction = weighted_prediction(p_unet_efficientnet, p_unet_seresnet)
    med_prediction = median_prediction(pixel_prediction, p_deeplab, p_pspnet, i)

    jaccard_score_true_class_unet_weighted += K.eval(jaccard_score_true_class(y,
        tf.keras.utils.to_categorical(pixel_prediction, num_classes=2).astype('float64')))
    jaccard_score_unet_weighted += K.eval(jaccard_score(y,
        tf.keras.utils.to_categorical(pixel_prediction, num_classes=2).astype('float64')))
    dice_score_true_class_unet_weighted += K.eval(dice_coef_true_class(y,
        tf.keras.utils.to_categorical(pixel_prediction, num_classes=2).astype('float64')))
    dice_score_unet_weighted += K.eval(dice_coef(y,
        tf.keras.utils.to_categorical(pixel_prediction, num_classes=2).astype('float64')))
    precision_unet_weighted += K.eval(sm.metrics.precision(y,
        tf.keras.utils.to_categorical(pixel_prediction, num_classes=2).astype('float64')))
    recall_unet_weighted += K.eval(sm.metrics.recall(y,
        tf.keras.utils.to_categorical(pixel_prediction, num_classes=2).astype('float64')))
    f2_score_unet_weighted += K.eval(sm.metrics.f2_score(y,
        tf.keras.utils.to_categorical(pixel_prediction, num_classes=2).astype('float64')))

    jaccard_score_true_class_median_prediction += K.eval(jaccard_score_true_class(y,
        tf.keras.utils.to_categorical(med_prediction, num_classes=2).astype('float64')))
    jaccard_score_median_prediction += K.eval(jaccard_score(y,
        tf.keras.utils.to_categorical(med_prediction, num_classes=2).astype('float64')))
    dice_score_true_class_median_prediction += K.eval(dice_coef_true_class(y,
        tf.keras.utils.to_categorical(med_prediction, num_classes=2).astype('float64')))
    dice_score_median_prediction += K.eval(dice_coef(y,
        tf.keras.utils.to_categorical(med_prediction, num_classes=2).astype('float64')))
    precision_median_prediction += K.eval(sm.metrics.precision(y,
        tf.keras.utils.to_categorical(med_prediction, num_classes=2).astype('float64')))
    recall_median_prediction += K.eval(sm.metrics.recall(y,
        tf.keras.utils.to_categorical(med_prediction, num_classes=2).astype('float64')))
    f2_score_median_prediction += K.eval(sm.metrics.f2_score(y,
        tf.keras.utils.to_categorical(med_prediction, num_classes=2).astype('float64')))


    p_unet_efficientnet = np.argmax(p_unet_efficientnet, axis=-1)
    p_unet_seresnet = np.argmax(p_unet_seresnet, axis=-1)
    p_deeplab = np.argmax(p_deeplab, axis=-1)
    p_pspnet = np.argmax(p_pspnet, axis=-1)

    jaccard_score_true_class_unet_efficientnet += K.eval(jaccard_score_true_class(y,
        tf.keras.utils.to_categorical(p_unet_efficientnet[0,:,:], num_classes=2).astype('float64')))
    jaccard_score_unet_efficientnet += K.eval(jaccard_score(y,
        tf.keras.utils.to_categorical(p_unet_efficientnet[0,:,:], num_classes=2).astype('float64')))
    dice_score_true_class_unet_efficicentnet += K.eval(dice_coef_true_class(y,
        tf.keras.utils.to_categorical(p_unet_efficientnet[0,:,:], num_classes=2).astype('float64')))
    dice_score_unet_efficicentnet += K.eval(dice_coef(y,
        tf.keras.utils.to_categorical(p_unet_efficientnet[0,:,:], num_classes=2).astype('float64')))
    precision_unet_efficientnet += K.eval(sm.metrics.precision(y,
        tf.keras.utils.to_categorical(p_unet_efficientnet[0,:,:], num_classes=2).astype('float64')))
    recall_unet_efficientnet += K.eval(sm.metrics.recall(y,
        tf.keras.utils.to_categorical(p_unet_efficientnet[0,:,:], num_classes=2).astype('float64')))
    f2_score_unet_efficientnet += K.eval(sm.metrics.f2_score(y,
        tf.keras.utils.to_categorical(p_unet_efficientnet[0,:,:], num_classes=2).astype('float64')))

    p_unet_efficientnet = np.expand_dims(p_unet_efficientnet, axis=-1)
    p_unet_efficientnet = p_unet_efficientnet * (255 / num_classes)
    p_unet_efficientnet = p_unet_efficientnet.astype(np.bool)

    #p_unet_efficientnet = np.concatenate([p_unet_efficientnet, p_unet_efficientnet, p_unet_efficientnet], axis=3)


    jaccard_score_true_class_unet_seresnet += K.eval(jaccard_score_true_class(y,
        tf.keras.utils.to_categorical(p_unet_seresnet[0,:,:], num_classes=2).astype('float64')))
    jaccard_score_unet_seresnet += K.eval(jaccard_score(y,
        tf.keras.utils.to_categorical(p_unet_seresnet[0,:,:], num_classes=2).astype('float64')))
    dice_score_true_class_unet_seresnet += K.eval(dice_coef_true_class(y,
        tf.keras.utils.to_categorical(p_unet_seresnet[0,:,:], num_classes=2).astype('float64')))
    dice_score_unet_seresnet += K.eval(dice_coef(y,
        tf.keras.utils.to_categorical(p_unet_seresnet[0,:,:], num_classes=2).astype('float64')))
    precision_unet_seresnet += K.eval(sm.metrics.precision(y,
        tf.keras.utils.to_categorical(p_unet_seresnet[0,:,:], num_classes=2).astype('float64')))
    recall_unet_seresnet += K.eval(sm.metrics.recall(y,
        tf.keras.utils.to_categorical(p_unet_seresnet[0,:,:], num_classes=2).astype('float64')))
    f2_score_unet_seresnet += K.eval(sm.metrics.f2_score(y,
        tf.keras.utils.to_categorical(p_unet_seresnet[0,:,:], num_classes=2).astype('float64')))

    jaccard_score_true_class_deeplab += K.eval(jaccard_score_true_class(y,
        tf.keras.utils.to_categorical(p_deeplab[0,:,:], num_classes=2).astype('float64')))
    jaccard_score_deeplab += K.eval(jaccard_score(y,
        tf.keras.utils.to_categorical(p_deeplab[0,:,:], num_classes=2).astype('float64')))
    dice_score_true_class_deeplab += K.eval(dice_coef_true_class(y,
        tf.keras.utils.to_categorical(p_deeplab[0,:,:], num_classes=2).astype('float64')))
    dice_score_deeplab += K.eval(dice_coef(y,
        tf.keras.utils.to_categorical(p_deeplab[0,:,:], num_classes=2).astype('float64')))
    precision_deeplab += K.eval(sm.metrics.precision(y,
        tf.keras.utils.to_categorical(p_deeplab[0,:,:], num_classes=2).astype('float64')))
    recall_deeplab += K.eval(sm.metrics.recall(y,
        tf.keras.utils.to_categorical(p_deeplab[0,:,:], num_classes=2).astype('float64')))
    f2_score_deeplab += K.eval(sm.metrics.f2_score(y,
        tf.keras.utils.to_categorical(p_deeplab[0,:,:], num_classes=2).astype('float64')))

    jaccard_score_true_class_pspnet += K.eval(jaccard_score_true_class(y_psp,
        tf.keras.utils.to_categorical(p_pspnet[0,:,:], num_classes=2).astype('float64')))
    jaccard_score_pspnet += K.eval(jaccard_score(y_psp,
        tf.keras.utils.to_categorical(p_pspnet[0,:,:], num_classes=2).astype('float64')))
    dice_score_true_class_pspnet += K.eval(dice_coef_true_class(y_psp,
        tf.keras.utils.to_categorical(p_pspnet[0,:,:], num_classes=2).astype('float64')))
    dice_score_pspnet += K.eval(dice_coef(y_psp,
        tf.keras.utils.to_categorical(p_pspnet[0,:,:], num_classes=2).astype('float64')))
    precision_pspnet += K.eval(sm.metrics.precision(y_psp,
        tf.keras.utils.to_categorical(p_pspnet[0,:,:], num_classes=2).astype('float64')))
    recall_pspnet += K.eval(sm.metrics.recall(y_psp,
        tf.keras.utils.to_categorical(p_pspnet[0,:,:], num_classes=2).astype('float64')))
    f2_score_pspnet += K.eval(sm.metrics.f2_score(y_psp,
        tf.keras.utils.to_categorical(p_pspnet[0,:,:], num_classes=2).astype('float64')))

    p_unet_seresnet = np.expand_dims(p_unet_seresnet, axis=-1)
    p_unet_seresnet = p_unet_seresnet * (255 / num_classes)
    p_unet_seresnet = p_unet_seresnet.astype(np.bool)


    p_deeplab = np.expand_dims(p_deeplab, axis=-1)
    p_deeplab = p_deeplab * (255 / num_classes)
    p_deeplab = p_deeplab.astype(np.bool)

    p_pspnet = np.expand_dims(p_pspnet, axis=-1)
    p_pspnet = p_pspnet * (255 / num_classes)
    p_pspnet = p_pspnet.astype(np.bool)

    # fig, ax = plt.subplots(1,2)
    # ax[0].imshow(p_deeplab[0,:,:])
    # ax[1].imshow(patches_prediction[0,:,:])
    # plt.show()

    #p_unet_seresnet = np.concatenate([p_unet_seresnet, p_unet_seresnet, p_unet_seresnet], axis=3)

    #patches_prediction = np.expand_dims(patches_prediction, axis=0)

    # p_ensemble_bitwise_and_patches = np.bitwise_and(p_ensemble_bitwise_or, patches_prediction)

    y = np.argmax(y, axis=-1)
    y = np.expand_dims(y, axis=-1)
    y = y * (255/num_classes)
    y = y.astype(np.int32)
    y = np.concatenate([y, y, y], axis=2)
    plt.imsave(f'results/mask/image_{i}.png', med_prediction, cmap = 'gray')
    visualize(
        fig_name=i,
        path=os.getcwd(),
        image=denormalize(image_efficientnet.squeeze()),
        gt=y,
        unet_weighted=pixel_prediction,
        deeplab=p_deeplab[0, :, :, :],
        pspnet=p_pspnet[0, :, :, :],
        final_prediction=med_prediction,
    )

print(f'Jaccard score U-Net (EfficientNet): {jaccard_score_unet_efficientnet / ids}')
print(f'Jaccard score true class U-Net (EfficientNet): {jaccard_score_true_class_unet_efficientnet / ids}')
print(f'Dice score U-Net (EfficientNet): {dice_score_unet_efficicentnet / ids}')
print(f'Dice score true class U-Net (EfficientNet): {dice_score_true_class_unet_efficicentnet / ids}')
print(f'Precision of U-Net (EfficientNet): {precision_unet_efficientnet / ids}')
print(f'Recall of U-Net (EfficientNet): {recall_unet_efficientnet / ids}')
print(f'F2-Score of U-Net (EfficientNet): {f2_score_unet_efficientnet / ids}\n')

print(f'Jaccard score U-Net (SeresNet): {jaccard_score_unet_seresnet / ids}')
print(f'Jaccard score true class U-Net (SeresNet): {jaccard_score_true_class_unet_seresnet / ids}')
print(f'Dice score U-Net (SeresNet): {dice_score_unet_seresnet / ids}')
print(f'Dice score true class U-Net (SeresNet): {dice_score_true_class_unet_seresnet / ids}')
print(f'Precision of U-Net (SeresNet): {precision_unet_seresnet / ids}')
print(f'Recall of U-Net (SeresNet): {recall_unet_seresnet / ids}')
print(f'F2-Score of U-Net (SeresNet): {f2_score_unet_seresnet / ids}\n')

print(f'Jaccard score Deeplab: {jaccard_score_deeplab / ids}')
print(f'Jaccard score true class Deeplab: {jaccard_score_true_class_deeplab / ids}')
print(f'Dice score Deeplab: {dice_score_deeplab / ids}')
print(f'Dice score true class Deeplab: {dice_score_true_class_deeplab / ids}')
print(f'Precision of Deeplab: {precision_deeplab / ids}')
print(f'Recall of Deeplab: {recall_deeplab / ids}')
print(f'F2-Score of Deeplab: {f2_score_deeplab / ids}\n')

print(f'Jaccard score PspNet: {jaccard_score_pspnet / ids}')
print(f'Jaccard score true class PspNet: {jaccard_score_true_class_pspnet / ids}')
print(f'Dice score PspNet: {dice_score_pspnet / ids}')
print(f'Dice score true class PspNet: {dice_score_true_class_pspnet / ids}')
print(f'Precision of PspNet: {precision_pspnet / ids}')
print(f'Recall of PspNet: {recall_pspnet / ids}')
print(f'F2-Score of PspNet: {f2_score_pspnet / ids}\n')

print(f'Jaccard score weighted U-Net ensemble: {jaccard_score_unet_weighted / ids}')
print(f'Jaccard score true class weighted U-Net ensemble: {jaccard_score_true_class_unet_weighted / ids}')
print(f'Dice score weighted U-Net ensemble: {dice_score_unet_weighted / ids}')
print(f'Dice score true class weighted U-Net ensemble: {dice_score_true_class_unet_weighted / ids}')
print(f'Precision of weighted U-Net ensemble: {precision_unet_weighted / ids}')
print(f'Recall of weighted U-Net ensemble: {recall_unet_weighted / ids}')
print(f'F2-Score of weighted U-Net ensemble: {f2_score_unet_weighted / ids}\n')

print(f'Jaccard score final ensemble: {jaccard_score_median_prediction / ids}')
print(f'Jaccard score true class final ensemble: {jaccard_score_true_class_median_prediction / ids}')
print(f'Dice score final ensemble: {dice_score_median_prediction / ids}')
print(f'Dice score true class final ensemble: {dice_score_true_class_median_prediction / ids}')
print(f'Precision of final ensemble: {precision_median_prediction / ids}')
print(f'Recall of final ensemble: {recall_median_prediction / ids}')
print(f'F2-Score of final ensemble: {f2_score_median_prediction / ids}\n')