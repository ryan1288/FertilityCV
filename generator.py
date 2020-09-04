from typing import Any

import numpy as np

from tensorflow.keras.utils import Sequence
from skimage.io import imread
from skimage.transform import resize


class DataGenerator(Sequence):
    def __init__(self, list_IDs, dim=(32,32,32), batch_size=32, n_channels=1, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    # Calculates the number of batches per epoch
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    # Generates one batch of data
    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(list_IDs_temp)

        return x, y

    # Updates indices after each epoch
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    # Generates data containing batch_size number of samples X : (n_samples, *dim, n_channels)
    def __data_generation(self, list_IDs_temp):
        # Initialization
        x = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.uint8)
        y = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.bool)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            img = imread('Data/' + ID)
            img = np.expand_dims(resize(img, self.dim, mode='constant', preserve_range=True), axis=-1)
            x[i] = img

            # Store class
            mask = imread('Label/' + ID[:-6] + 'DAPI.png')
            mask = np.expand_dims(resize(mask, self.dim, mode='constant', preserve_range=True), axis=-1)
            y[i] = mask

        return x, y
