from tensorflow import keras
from Utils.metrics import jaccard_score, jaccard_score_true_class, dice_coef
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
BACKBONE = 'efficientnetb3'
optim = keras.optimizers.Adam(0.001)
preprocess_input = sm.get_preprocessing(BACKBONE)
num_classes = 2

DATA_DIR = f'{Path(__file__).parent.parent.parent}/Dataset'

x_test_dir = os.path.join(DATA_DIR, 'test_augmented/img')
y_test_dir = os.path.join(DATA_DIR, 'test_augmented/mask')
y_test_edges_dir = os.path.join(DATA_DIR, 'test_augmented/mask_edge')
test_len = len(os.listdir(x_test_dir))


test_dataset = Dataset(
    x_test_dir,
    y_test_dir,
    img_size = img_size,
    classes=CLASSES,
    preprocessing=get_preprocessing(preprocess_input),
)

test_dataset_edges = Dataset(
    x_test_dir,
    y_test_edges_dir,
    edges = True,
    img_size = img_size,
    classes=CLASSES,
)

test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False, train_len=test_len)
test_dataloader_edges = Dataloder(test_dataset_edges, batch_size=1, shuffle=False, train_len=test_len)

object_model_unet = keras.models.load_model(f'../U-Net/models/unet_1_efficientnetb3_jaccard_loss_256_size_cropped_25_epoch_.h5',
                                 custom_objects={'jaccard_loss': jaccard_loss, 'jaccard_score_true_class': jaccard_score_true_class,
                                                  'jaccard_score': jaccard_score,'dice_coef': dice_coef})

object_model_deeplab = keras.models.load_model(f'../Deeplab/models/deeplab_1_jaccard_loss_256_size_cropped_25_epoch_.h5',
                                 custom_objects={'jaccard_loss': jaccard_loss, 'jaccard_score_true_class': jaccard_score_true_class,
                                                  'jaccard_score': jaccard_score,'dice_coef': dice_coef})

object_models = [object_model_unet, object_model_deeplab]
object_model_unet._name = 'model_unet'
object_model_deeplab._name = 'model_deeplab'
# object_model_unet.get_layer(name='model').name = 'model_unet'
# object_model_deeplab.get_layer(name='model').name = 'model_deeplab'

model_input = keras.Input(shape=(256, 256, 3))
model_outputs = [object_model_unet(model_input), object_model_deeplab(model_input)]
ensemble_output = keras.layers.Average()(model_outputs)
ensemble_model = keras.Model(inputs=model_input, outputs=ensemble_output)
ensemble_model.compile(optim, jaccard_loss, metrics)

edge_model_unet = keras.models.load_model(f'../U-Net/models_for_edges/unet_1_efficientnetb3_modified_sum_jaccard_loss_256_size_cropped_only_edges_20_epoch_dilation_3_edge_enhanced.h5',
                                          custom_objects={'weighted_jaccard_loss': weighted_jaccard_loss, 'jaccard_score_true_class': jaccard_score_true_class,
                                                  'jaccard_score': jaccard_score,'dice_coef': dice_coef})

edge_model_deeplab = keras.models.load_model(f'../Deeplab/models_for_edges/deeplab_1_modified_sum_jaccard_loss_256_size_cropped_25_epoch_.h5',
                                          custom_objects={'weighted_jaccard_loss': weighted_jaccard_loss, 'jaccard_score_true_class': jaccard_score_true_class,
                                                  'jaccard_score': jaccard_score,'dice_coef': dice_coef})

# scores_object_unet = object_model_unet.evaluate(test_dataloader)
# scores_object_deeplab = object_model_deeplab.evaluate(test_dataloader)
# scores_object_ensemble = ensemble_model.evaluate(test_dataloader)
#scores_edges = edge_model.evaluate(test_dataloader_edges)

# for scores in [scores_object_unet, scores_object_deeplab, scores_object_ensemble]:
#     print("Loss: {:.5}".format(scores[0]))
#     for metric, value in zip(metrics, scores[1:]):
#         print("mean {}: {:.5}".format(metric.__name__, value))

ids = len(test_dataloader)
for i in range(10):
    num = np.random.randint(0,ids)
    image, y = test_dataset[num]
    image_edge, y_edges = test_dataset_edges[num]

    image = np.expand_dims(image, axis=0)
    image_edge = np.expand_dims(image_edge, axis=0)

    y = np.argmax(y, axis=-1)
    y = np.expand_dims(y, axis=-1)
    y = y * (255/num_classes)
    y = y.astype(np.int32)
    y = np.concatenate([y, y, y], axis=2)

    y_edges = np.argmax(y_edges, axis=-1)
    y_edges = np.expand_dims(y_edges, axis=-1)
    y_edges = y_edges * (255/num_classes)
    y_edges = y_edges.astype(np.int32)
    y_edges = np.concatenate([y_edges, y_edges, y_edges], axis=2)

    p_unet = object_model_unet.predict(image)
    p_unet = np.argmax(p_unet, axis=-1)
    p_unet = np.expand_dims(p_unet, axis=-1)
    p_unet = p_unet * (255/num_classes)
    p_unet = p_unet.astype(np.int32)
    p_unet = np.concatenate([p_unet, p_unet, p_unet], axis=3)
    #p = p * (255/num_classes)
    #p = p.astype(np.int32)
    #p = np.concatenate([p, p, p], axis=2)

    p_deeplab = object_model_deeplab.predict(image)
    p_deeplab = np.argmax(p_deeplab, axis=-1)
    p_deeplab = np.expand_dims(p_deeplab, axis=-1)
    p_deeplab = p_deeplab * (255/num_classes)
    p_deeplab = p_deeplab.astype(np.int32)
    p_deeplab = np.concatenate([p_deeplab, p_deeplab, p_deeplab], axis=3)
    #p = p * (255/num_classes)
    #p = p.astype(np.int32)
    #p = np.concatenate([p, p, p], axis=2)

    p_ensemble = ensemble_model.predict(image)
    p_ensemble = np.argmax(p_ensemble, axis=-1)
    p_ensemble = np.expand_dims(p_ensemble, axis=-1)
    p_ensemble = p_ensemble * (255/num_classes)
    p_ensemble = p_ensemble.astype(np.int32)
    p_ensemble = np.concatenate([p_ensemble, p_ensemble, p_ensemble], axis=3)
    #p = p * (255/num_classes)
    #p = p.astype(np.int32)
    #p = np.concatenate([p, p, p], axis=2)

    p_edges_unet = edge_model_unet.predict(image_edge)
    p_edges_unet = np.argmax(p_edges_unet, axis=-1)
    p_edges_unet = np.expand_dims(p_edges_unet, axis=-1)
    p_edges_unet = p_edges_unet * (255 / num_classes)
    p_edges_unet = p_edges_unet.astype(np.int32)
    p_edges_unet = np.concatenate([p_edges_unet, p_edges_unet, p_edges_unet], axis=3)

    p_edges_deeplab = edge_model_deeplab.predict(image_edge)
    p_edges_deeplab = np.argmax(p_edges_deeplab, axis=-1)
    p_edges_deeplab = np.expand_dims(p_edges_deeplab, axis=-1)
    p_edges_deeplab = p_edges_deeplab * (255 / num_classes)
    p_edges_deeplab = p_edges_deeplab.astype(np.int32)
    p_edges_deeplab = np.concatenate([p_edges_deeplab, p_edges_deeplab, p_edges_deeplab], axis=3)

    visualize(
        fig_name=i,
        path=None,
        image=denormalize(image.squeeze()),
        image_edge=denormalize(image_edge.squeeze()),
        gt_mask=y,
        gt_mask_edges=y_edges,
        pr_mask_unet=p_unet[0,:,:,:],
        pr_mask_deeplab=p_deeplab[0,:,:,:],
        pr_mask_ensemble=p_ensemble[0,:,:,:],
        pr_mask_edges_unet=p_edges_unet[0, :, :, :],
        pr_mask_edges_deeplab=p_edges_deeplab[0, :, :, :],
    )

