import matplotlib.pyplot as plt
import os
import random
import tensorflow as tf
import cv2
import numpy as np
import keras.backend as K
from Utils.losses import jaccard_loss, weighted_jaccard_loss
from Utils.metrics import jaccard_score, jaccard_score_true_class, dice_coef
from pathlib import Path
from Utils.augmentation import get_training_augmentation, get_validation_augmentation, get_preprocessing
from Utils.dataset import Dataset, Dataloder
from Utils.visuals import visualize, denormalize, visualize_edges

img_size = (256, 256)
num_classes = 2
batch_size = 4


DATA_DIR = f'{Path(__file__).parent.parent.parent}/Dataset'

x_train_dir = os.path.join(DATA_DIR, 'training_augmented/img')
y_train_dir = os.path.join(DATA_DIR, 'training_augmented/mask_edge')
train_len = len(os.listdir(x_train_dir))

x_valid_dir = os.path.join(DATA_DIR, 'validation_augmented/img')
y_valid_dir = os.path.join(DATA_DIR, 'validation_augmented/mask_edge')
valid_len = len(os.listdir(x_valid_dir))

x_test_dir = os.path.join(DATA_DIR, 'test_augmented/img')
y_test_dir = os.path.join(DATA_DIR, 'test_augmented/mask_edge')
test_len = len(os.listdir(x_test_dir))

from tensorflow.keras import layers
from tensorflow import keras
import segmentation_models as sm

def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

def DeeplabV3Plus(image_size, num_classes):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    resnet50 = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same", activation = 'sigmoid')(x)
    return keras.Model(inputs=model_input, outputs=model_output)


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

BACKBONE = 'resnet18'
BATCH_SIZE = batch_size
CLASSES = ['background', 'polyp']
LR = 0.0001
EPOCHS = 10

preprocess_input = sm.get_preprocessing(BACKBONE)

activation = 'sigmoid'

#create model
#model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
model = DeeplabV3Plus(img_size[0], num_classes)

optim = tf.keras.optimizers.Adam(LR)
# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
#dice_loss = sm.losses.DiceLoss(class_weights=[0.5, 2, 2, 2])


total_loss = weighted_jaccard_loss

# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss

metrics = [jaccard_score_true_class,jaccard_score, dice_coef]

model.summary()
model.compile(optim, total_loss, metrics)
# compile keras model with defined optimozer, loss and metrics

train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    edges=True,
    img_size = img_size,
    classes=CLASSES,
    augmentation=get_training_augmentation(img_size),
    #preprocessing=get_preprocessing(preprocess_input),
)

image, y = train_dataset[0]
fig, ax = plt.subplots(1,2)
ax[0].imshow(image)
ax[1].imshow(y[:,:,1])
plt.show()

# Dataset for validation images
valid_dataset = Dataset(
    x_valid_dir,
    y_valid_dir,
    edges=True,
    img_size=img_size,
    classes=CLASSES,
    augmentation=get_validation_augmentation(img_size),
    #preprocessing=get_preprocessing(preprocess_input),
)

train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True, length=train_len // batch_size, train_len=train_len)
valid_dataloader = Dataloder(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, length=valid_len // batch_size, validation=True, validation_len=valid_len)

#model desc
model_num = 1
loss = 'modified_sum_jaccard_loss'
image_size = str(img_size[0])
image_mode = 'cropped'
add_info = 'mod_preprocess_2'
model_name = f'deeplab_{model_num}_{loss}_{image_size}_size_{image_mode}_{BACKBONE}_{EPOCHS}_epoch_{add_info}'

# define callbacks for learning rate scheduling and best checkpoints saving
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(f"models_for_edges/{model_name}.h5", save_best_only=True, mode='min'),
    tf. keras.callbacks.ReduceLROnPlateau(),
    #tf.keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=0,patience=10,verbose=0,mode="auto",baseline=None,restore_best_weights=False,)
]

# model = keras.models.load_model(f'models_for_edges/{model_name}.h5',
#                                  custom_objects={'weighted_jaccard_loss': weighted_jaccard_loss, 'jaccard_score_true_class': jaccard_score_true_class,
#                                                   'jaccard_score': jaccard_score,'dice_coef': dice_coef})
#
# model_name = f'deeplab_{model_num}_{loss}_{image_size}_size_{image_mode}_{BACKBONE}_{EPOCHS + 10}_epoch_{add_info}'

# history = model.fit(
#     train_dataloader,
#     steps_per_epoch=len(train_dataloader),
#     epochs=EPOCHS,
#     callbacks=callbacks,
#     validation_data=valid_dataloader,
#     validation_steps=len(valid_dataloader),
# )
#
# # Plot training & validation iou_score values
# fig, ax = plt.subplots(2,2)
#
# ax[0,0].plot(history.history["loss"])
# ax[0,0].set_title("Training Loss")
# ax[0,0].set_ylabel("loss")
# ax[0,0].set_xlabel("epoch")
#
# ax[0,1].plot(history.history["jaccard_score_true_class"])
# ax[0,1].set_title("Training Accuracy")
# ax[0,1].set_ylabel("accuracy")
# ax[0,1].set_xlabel("epoch")
#
# ax[1,0].plot(history.history["val_loss"])
# ax[1,0].set_title("Validation Loss")
# ax[1,0].set_ylabel("val_loss")
# ax[1,0].set_xlabel("epoch")
#
# ax[1,1].plot(history.history["val_jaccard_score_true_class"])
# ax[1,1].set_title("Validation Accuracy")
# ax[1,1].set_ylabel("val_accuracy")
# ax[1,1].set_xlabel("epoch")
# plt.savefig(f'results_for_edges/plots/model_plot_{model_name}.png', dpi = 300)

test_dataset = Dataset(
    x_test_dir,
    y_test_dir,
    edges=True,
    img_size=img_size,
    classes=CLASSES,
    #augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
)

test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False, train_len=test_len)

model = keras.models.load_model(f'models_for_edges/{model_name}.h5',
                                 custom_objects={'weighted_jaccard_loss': weighted_jaccard_loss, 'jaccard_score_true_class': jaccard_score_true_class,
                                                  'jaccard_score': jaccard_score,'dice_coef': dice_coef})

scores = model.evaluate(test_dataloader)

f = open(f'results_for_edges/metrics/{model_name}.txt','w')
print("Loss: {:.5}".format(scores[0]), file=f)
for metric, value in zip(metrics, scores[1:]):
    print("mean {}: {:.5}".format(metric.__name__, value), file=f)
f.close()

n = 5
ids = len(test_dataloader)

for i in range(10):
    image, y = test_dataset[np.random.randint(0,ids)]
    image = np.expand_dims(image, axis=0)
    y = np.argmax(y, axis=-1)
    y = np.expand_dims(y, axis=-1)
    y = y * (255/num_classes)
    y = y.astype(np.int32)
    y = np.concatenate([y, y, y], axis=2)

    p = model.predict(image)
    p = np.argmax(p, axis=-1)
    p = np.expand_dims(p, axis=-1)
    p = p * (255/num_classes)
    p = p.astype(np.int32)
    p = np.concatenate([p, p, p], axis=3)

    visualize_edges(
        fig_name=i,
        path= os.getcwd(),
        image=denormalize(image.squeeze()),
        gt_mask=y,
        pr_mask=p[0,:,:,:],
    )