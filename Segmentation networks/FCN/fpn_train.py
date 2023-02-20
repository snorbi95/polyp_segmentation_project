import matplotlib.pyplot as plt
import os
import random
import tensorflow as tf
import cv2
import numpy as np
from Utils.losses import jaccard_loss
from Utils.metrics import jaccard_score, jaccard_score_true_class, dice_coef
from Utils.augmentation import get_training_augmentation, get_validation_augmentation, get_preprocessing
from Utils.dataset import Dataset, Dataloder
from Utils.visuals import visualize, denormalize
from pathlib import Path

img_size = (128, 128)
num_classes = 2
batch_size = 8


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


import segmentation_models as sm

BACKBONE = 'efficientnetb3'
BATCH_SIZE = batch_size
CLASSES = ['background', 'polyp']
LR = 0.001
EPOCHS = 15

preprocess_input = sm.get_preprocessing(BACKBONE)

n_classes = 2  # case for binary and multiclass segmentation
activation = 'sigmoid'

#create model
model = sm.FPN(BACKBONE, input_shape = (img_size[0], img_size[1], 3), classes=n_classes, activation=activation)

optim = tf.keras.optimizers.Adam(LR)

total_loss = jaccard_loss


metrics = [jaccard_score_true_class, jaccard_score, dice_coef]
model.summary()
# compile keras model with defined optimozer, loss and metrics
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
valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False, length=valid_len // batch_size, validation=True, validation_len=valid_len)

model_num = 1
loss = 'jaccard_loss'
image_size = str(img_size[0])
image_mode = 'augmented_patches'
add_info = ''
model_name = f'fpn_{model_num}_{loss}_{image_size}_size_{image_mode}_{BACKBONE}_{EPOCHS}_epoch_{add_info}'

# define callbacks for learning rate scheduling and best checkpoints saving
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(f"models/{model_name}.h5", save_best_only=True, mode='min'),
    tf. keras.callbacks.ReduceLROnPlateau(),
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
plt.savefig(f'results/model_plot_{model_name}.png', dpi = 300)

test_dataset = Dataset(
    x_test_dir,
    y_test_dir,
    img_size=img_size,
    classes=CLASSES,
    augmentation=get_validation_augmentation(img_size),
    preprocessing=get_preprocessing(preprocess_input),
)

model = tf.keras.models.load_model(f'models/{model_name}.h5',
                                 custom_objects={'jaccard_loss': jaccard_loss, 'jaccard_score_true_class': jaccard_score_true_class,
                                                  'jaccard_score': jaccard_score,'dice_coef': dice_coef})

test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False, train_len=test_len)

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
        path = os.getcwd(),
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
