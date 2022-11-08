import math

import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
import numpy as np
import keras.backend as K
from Utils.metrics import dice_coef, jaccard_score, jaccard_score_true_class
from Utils.losses import jaccard_loss
from Utils.augmentation import get_training_augmentation, get_validation_augmentation, get_preprocessing
from Utils.dataset import Dataset, Dataloder
from Utils.visuals import visualize, denormalize
from pathlib import Path

img_size = (224, 224)
num_classes = 2
batch_size = 4


DATA_DIR = f'{Path(__file__).parent.parent.parent}/Dataset'

x_train_dir = os.path.join(DATA_DIR, 'training_augmented/img')
y_train_dir = os.path.join(DATA_DIR, 'training_augmented/mask')
train_len = len(os.listdir(x_train_dir))

x_valid_dir = os.path.join(DATA_DIR, 'validation_augmented/img')
y_valid_dir = os.path.join(DATA_DIR, 'validation_augmented/mask')
valid_len = len(os.listdir(x_valid_dir))

x_test_dir = os.path.join(DATA_DIR, 'test_augmented/img')
y_test_dir = os.path.join(DATA_DIR, 'test_augmented/mask')
test_len = len(os.listdir(x_test_dir))


from tensorflow.keras import layers
from tensorflow import keras
import segmentation_models as sm

def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()
BACKBONE = 'efficientnetb3'
BATCH_SIZE = batch_size
CLASSES = ['background', 'polyp']
LR = 0.001
EPOCHS = 15

preprocess_input = sm.get_preprocessing(BACKBONE)

n_classes = 2
activation = 'sigmoid'

#create model
model = sm.Linknet(BACKBONE, classes=n_classes, activation=activation)
#model = get_model(img_size, num_classes)

optim = tf.keras.optimizers.Adam(LR)

metrics = [jaccard_score_true_class, jaccard_score, dice_coef]

model.compile(optim, jaccard_loss, metrics)
# compile keras model with defined optimozer, loss and metrics

model.summary()

train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    img_size = img_size,
    classes=CLASSES,
    augmentation=get_training_augmentation(img_size),
    preprocessing=get_preprocessing(preprocess_input),
)

# Dataset for validation images
valid_dataset = Dataset(
    x_valid_dir,
    y_valid_dir,
    img_size = img_size,
    classes=CLASSES,
    augmentation=get_validation_augmentation(img_size),
    preprocessing=get_preprocessing(preprocess_input),
)

train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True, length=train_len // batch_size, train_len=train_len)
valid_dataloader = Dataloder(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, length=valid_len // batch_size, validation=True, validation_len=valid_len)

#model desc
model_num = 1
loss = 'jaccard_loss'
image_size = str(img_size[0])
image_mode = 'cropped'
add_info = f''
model_name = f'linknet_{model_num}_{BACKBONE}_{loss}_{image_size}_size_{image_mode}_{EPOCHS}_epoch_{add_info}'
# model = keras.models.load_model(f'models/unet_1_efficientnetb3_dice_loss_plus_focal_loss_224_size_crop_6_epoch_binary_arthery_w_true_weight_0.1_nadam.h5',
#                                 custom_objects={'jaccard_loss': jaccard_loss,
#                                                 'jaccard_score_true_class': jaccard_score_true_class,
#                                                 'jaccard_score_all': jaccard_score_all,
#                                                 'dice_coef': dice_coef})
# define callbacks for learning rate scheduling and best checkpoints saving
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(f"models/{model_name}.h5", save_best_only=True, mode='min'),
    tf. keras.callbacks.ReduceLROnPlateau(),
    #tf.keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=0,patience=10,verbose=0,mode="auto",baseline=None,restore_best_weights=False,)
]

history = model.fit(
    train_dataloader,
    steps_per_epoch=len(train_dataloader),
    epochs=EPOCHS,
    callbacks=callbacks,
    validation_data=valid_dataloader,
    validation_steps=len(valid_dataloader),
)

#Plot training & validation iou_score values
fig, ax = plt.subplots(2,2)

ax[0,0].plot(history.history["loss"])
ax[0,0].set_title("Training Loss")
ax[0,0].set_ylabel("loss")
ax[0,0].set_xlabel("epoch")

ax[0,1].plot(history.history["jaccard_score"])
ax[0,1].set_title("Training Accuracy")
ax[0,1].set_ylabel("accuracy")
ax[0,1].set_xlabel("epoch")

ax[1,0].plot(history.history["val_loss"])
ax[1,0].set_title("Validation Loss")
ax[1,0].set_ylabel("val_loss")
ax[1,0].set_xlabel("epoch")

ax[1,1].plot(history.history["val_jaccard_score"])
ax[1,1].set_title("Validation Accuracy")
ax[1,1].set_ylabel("val_accuracy")
ax[1,1].set_xlabel("epoch")
plt.savefig(f'results/plots/model_plot_{model_name}.png', dpi = 300)

test_dataset = Dataset(
    x_test_dir,
    y_test_dir,
    img_size = img_size,
    classes=CLASSES,
    #augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
)

test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False, train_len=test_len)

model = keras.models.load_model(f'models/{model_name}.h5',
                                 custom_objects={'jaccard_loss': jaccard_loss, 'jaccard_score_true_class': jaccard_score_true_class,
                                                  'jaccard_score': jaccard_score,'dice_coef': dice_coef})
scores = model.evaluate(test_dataloader)

f = open(f'results/metrics/{model_name}.txt','w')
print("Loss: {:.5}".format(scores[0]), file=f)
for metric, value in zip(metrics, scores[1:]):
    print("mean {}: {:.5}".format(metric.__name__, value), file=f)
f.close()

n = 5
ids = len(test_dataloader)
image, y = test_dataset[np.random.randint(0, ids)]
# plt.imshow(denormalize(image))
# plt.show()

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

    visualize(
        fig_name=i,
        path=os.getcwd(),
        image=denormalize(image.squeeze()),
        gt_mask=y,
        pr_mask=p[0,:,:,:],
    )
# data_gen_args = dict(rescale=1./255,)
# image_datagen = ImageDataGenerator(**data_gen_args)
# mask_datagen = ImageDataGenerator(**data_gen_args)
# # image_datagen.fit(images)
# # mask_datagen.fit(masks)
# # Provide the same seed and keyword arguments to the fit and flow methods
# seed = 1
# image_generator = image_datagen.flow_from_directory(
#     '../dataset/train_gen/images',
#     batch_size=batch_size,
#     class_mode=None,
#     # color_mode='grayscale',
#     seed=seed)
# mask_generator = mask_datagen.flow_from_directory(
#     '../dataset/train_gen/mask',
#     batch_size=batch_size,
#     class_mode=None,
#     color_mode='grayscale',
#     seed=seed)
# # combine generators into one which yields image and masks
# train_generator = zip(image_generator, mask_generator)
#
# image_test_generator = image_datagen.flow_from_directory(
#     '../dataset/test_gen/images',
#     batch_size=batch_size,
#     class_mode=None,
#     # color_mode='grayscale',
#     seed=seed)
# mask_test_generator = mask_datagen.flow_from_directory(
#     '../dataset/test_gen/mask',
#     batch_size=batch_size,
#     class_mode=None,
#     color_mode='grayscale',
#     seed=seed)
# # combine generators into one which yields image and masks
# val_generator = zip(image_test_generator, mask_test_generator)
