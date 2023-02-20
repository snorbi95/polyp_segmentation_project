import os
import cv2
import tensorflow as tf
import numpy as np

# classes for data loading and preprocessing
from matplotlib import pyplot as plt


class Dataset:
    CLASSES = ['background', 'polyp']

    def __init__(
            self,
            images_dir,
            masks_dir,
            img_size,
            edges=None,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.img_size = img_size
        self.edges = edges
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i], cv2.IMREAD_COLOR)
        image = cv2.resize(image, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_NEAREST)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.edges:
            from skimage import exposure, filters, color, feature, morphology
            image = color.rgb2hsv(image)
            _, gray_image, _ = cv2.split(image)
            image = color.hsv2rgb(image)
            r, g, b = cv2.split(image)
            #gray_image = color.rgb2gray(image)
            gray_image = exposure.adjust_sigmoid(gray_image, cutoff=0.5)
            gray_image = exposure.equalize_hist(gray_image)
            #gray_image = exposure.rescale_intensity(gray_image, out_range=(0,1))
            #gray_image = 1 - gray_image
            r = exposure.rescale_intensity(r * gray_image, out_range=(0,255)).astype(np.uint8)
            g = exposure.rescale_intensity(g * gray_image, out_range=(0,255)).astype(np.uint8)
            b = exposure.rescale_intensity(b * gray_image, out_range=(0,255)).astype(np.uint8)

        #print(self.masks_fps[i])
        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_NEAREST)
        mask[mask == 255] = 1
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(gray_image)
        # ax[1].imshow(image)
        # plt.show()
        #image = color.hsv2rgb(image)
        #image = cv2.merge((r, g, b))
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # # add background if mask is not binary
        # if mask.shape[-1] != 1:
        #     background = 1 - mask.sum(axis=-1, keepdims=True)
        #     mask = np.concatenate((mask, background), axis=-1)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


class Dataloder(tf.keras.utils.Sequence):
    """Load data from dataset and form batches

    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(self, dataset, batch_size=4, shuffle=False, length = None, train_len = 1, validation = None, validation_len = 1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))
        self.length = length
        self.train_len = train_len
        self.validation = validation
        self.validation_len = validation_len
        self.on_epoch_end()

    def __getitem__(self, i):

        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            if self.validation:
                data.append(self.dataset[j % self.validation_len])
            else:
                data.append(self.dataset[j % self.train_len])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        if self.length == None:
            return len(self.indexes) // self.batch_size
        else:
            return self.length

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
