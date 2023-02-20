import keras
from keras.layers import Conv2D, Dropout, Conv2DTranspose, Add
from keras.activations import sigmoid
from keras.layers.core import Activation
from keras.layers.convolutional import Cropping2D
from keras.models import Model
from keras.utils import plot_model

from keras.layers import Conv2D, Dropout, Input, MaxPooling2D
import keras

from keras import backend as K

from pylab import *
from keras.layers import *

from keras.models import *

from keras.layers import MaxPooling2D, Cropping2D, Conv2D
from keras.layers import Input, Add, Dropout
import cv2

import segmentation_models as sm
import tensorflow as tf
from Utils.losses import jaccard_loss
from Utils.metrics import jaccard_score, jaccard_score_true_class, dice_coef
import os
from pathlib import Path
from Utils.augmentation import get_training_augmentation, get_validation_augmentation, get_preprocessing
from Utils.dataset import Dataset, Dataloder
from Utils.visuals import visualize, denormalize

img_size = (128, 128)
num_classes = 2
batch_size = 4

DATA_DIR = f'{Path(__file__).parent.parent.parent}/Dataset'

x_train_dir = os.path.join(DATA_DIR, 'training_augmented_small/img')
y_train_dir = os.path.join(DATA_DIR, 'training_augmented_small/mask')
train_len = len(os.listdir(x_train_dir))

x_valid_dir = os.path.join(DATA_DIR, 'validation_augmented_small/img')
y_valid_dir = os.path.join(DATA_DIR, 'validation_augmented_small/mask')
valid_len = len(os.listdir(x_valid_dir))

x_test_dir = os.path.join(DATA_DIR, 'test_augmented_small/img')
y_test_dir = os.path.join(DATA_DIR, 'test_augmented_small/mask')
test_len = len(os.listdir(x_test_dir))


def bilinear(shape, dtype=None):
    filter_size = shape[0]
    num_channels = shape[2]

    bilinear_kernel = np.zeros([filter_size, filter_size], dtype=np.float32)
    scale_factor = (filter_size + 1) // 2
    if filter_size % 2 == 1:
        center = scale_factor - 1
    else:
        center = scale_factor - 0.5
    for x in range(filter_size):
        for y in range(filter_size):
            bilinear_kernel[x, y] = (1 - abs(x - center) / scale_factor) * \
                                    (1 - abs(y - center) / scale_factor)
    weights = np.zeros((filter_size, filter_size, num_channels, num_channels))
    for i in range(num_channels):
        weights[:, :, i, i] = bilinear_kernel

    return weights

def vgg_encoder( shape ):

    img_input = Input(shape)

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1' )(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2' )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool' )(x)
    f1 = x
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1' )(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2' )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool' )(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1' )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2' )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3' )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool' )(x)
    f3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1' )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2' )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3' )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool' )(x)
    f4 = x

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1' )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2' )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3' )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool' )(x)
    f5 = x

    return img_input , [f3 , f4 , f5 ]


class fcn8(object):

    def __init__(self, n_classes, shape):
        self.n_classes = n_classes
        self.shape = shape

    def get_model(self):
        n_classes = self.n_classes
        shape = self.shape

        img_input, [f3, f4, f5] = vgg_encoder(shape)

        o = f5
        o = Conv2D(2048, (7, 7), activation='relu', padding='same')(o)
        o = Dropout(0.5)(o)
        o = Conv2D(1024, (1, 1), activation='relu', padding='same')(o)
        o = Dropout(0.5)(o)

        o = Conv2D(n_classes, (1, 1), activation='relu')(o)
        o = Conv2DTranspose(n_classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False, kernel_initializer=bilinear)(
            o)
        o = Cropping2D(((1, 1), (1, 1)))(o)

        o2 = f4
        o2 = Conv2D(n_classes, (1, 1), activation='relu')(o2)

        o = Add()([o, o2])
        o = Conv2DTranspose(n_classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(o)
        o = Cropping2D(((1, 1), (1, 1)))(o)

        o2 = f3
        o2 = Conv2D(n_classes, (1, 1), activation='relu')(o2)

        o = Add()([o2, o])
        o = Conv2DTranspose(n_classes, kernel_size=(16, 16), strides=(8, 8), use_bias=False)(o)
        o = Cropping2D(((4, 4), (4, 4)))(o)
        o = Activation('sigmoid')(o)

        model = Model(img_input, o)
        model.model_name = "fcn_8"
        return model


BACKBONE = 'efficientnetb3'
BATCH_SIZE = batch_size
CLASSES = ['background', 'polyp']
LR = 0.001
EPOCHS = 15

preprocess_input = sm.get_preprocessing(BACKBONE)
activation = 'sigmoid'

x = fcn8(shape=(img_size[0], img_size[1], 3), n_classes=num_classes)
model = x.get_model()
model.summary()

optim = tf.keras.optimizers.Adam(LR)
total_loss = jaccard_loss
metrics = [jaccard_score_true_class,jaccard_score, dice_coef]

model.compile(optim, total_loss, metrics)

train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    img_size=img_size,
    classes=CLASSES,
    augmentation=get_training_augmentation(img_size),
    preprocessing=get_preprocessing(preprocess_input),
)

# Dataset for validation images
valid_dataset = Dataset(
    x_valid_dir,
    y_valid_dir,
    img_size=img_size,
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
image_mode = 'cropped_patches'
add_info = ''
model_name = f'fcn_{model_num}_{loss}_{image_size}_size_{image_mode}_{EPOCHS}_epoch_{add_info}'

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

# Plot training & validation iou_score values
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
    img_size=img_size,
    classes=CLASSES,
    #augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
)

test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)

scores = model.evaluate(test_dataloader)

f = open(f'results/metrics/{model_name}.txt','w')
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

    visualize(
        fig_name=i,
        path=os.getcwd(),
        image=denormalize(image.squeeze()),
        gt_mask=y,
        pr_mask=p[0,:,:,:],
    )