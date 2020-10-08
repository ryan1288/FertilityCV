import os
import numpy as np
import tensorflow as tf

from tqdm import tqdm  # progress bars on database extraction
from skimage.io import imread  # show images as windows
from skimage.transform import resize  # resize images


# Purpose: Access directories to extract, resize, and transform datasets into numpy arrays
# Parameters:
#   data_path: dataset directory address (string)
#   label_path: label directory address (string)
#   height: image height (int)
#   width: image width (int)
#   channels: image channels (int)
def create_train_arrays(image_path, label_path, height, width, channels):
    labels = os.listdir(label_path)

    # Initialize numpy arrays to store both the color image and mask
    x_set = np.zeros((len(labels), height, width, channels), dtype=np.uint8)
    y_set = np.zeros((len(labels), height, width, 1), dtype=np.bool)

    # A set of unique boolean mask images for each object instance
    i = 0
    for mask in tqdm(labels):
        path = label_path + mask
        img = imread(path)
        img = np.expand_dims(resize(img, (height, width), mode='constant', preserve_range=True), axis=-1)
        y_set[i] = img

        image = mask[:-8] + 'BF.png'
        path = image_path + image
        img = imread(path)
        img = np.expand_dims(resize(img, (height, width), mode='constant', preserve_range=True), axis=-1)
        x_set[i] = img
        i += 1

    return x_set, y_set


# Purpose: Access directories to extract, resize, and transform datasets into numpy arrays
# Parameters:
#   data_path: dataset directory address (string)
#   height: image height (int)
#   width: image width (int)
#   channels: image channels (int)
def create_test_arrays(data_path, height, width, channels):
    # Assign image ids through the directory
    image_ids = next(os.walk(data_path))[1]

    # Initialize numpy arrays to store both the color image and mask
    x_set = np.zeros((len(image_ids), height, width, channels), dtype=np.uint8)

    # Loop through every image using given path and unique folder identifier
    for i, id_ in tqdm(enumerate(image_ids), total=len(image_ids)):
        path = data_path + id_
        img = imread(path + '/images/' + id_ + '.png')[:, :, :channels]
        img = resize(img, (height, width), mode='constant', preserve_range=True)
        x_set[i] = img

    return x_set


# Purpose: Create data augmented generators that include the train-validation split for model training
# Parameters:
#   x_train: input training data
#   y_train: output mask ground truth
#   valid_split: % of training data for validation
#   batch_size: size of batch when training
def create_generators(x_train, y_train, valid_split, batch_size):
    # Seeds for consistent runs for debugging (optional)
    seed = 47  # Arbitrary value

    # Create generators that will augment the data for each epoch

    image_gen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=45, zoom_range=0.2,
                                                                width_shift_range=0.4, height_shift_range=0.4,
                                                                brightness_range=(0.8, 1.2), fill_mode='reflect',
                                                                horizontal_flip=True, vertical_flip=True)
    mask_gen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=45, zoom_range=0.2,
                                                               width_shift_range=0.4, height_shift_range=0.4,
                                                               brightness_range=(0.8, 1.2), fill_mode='reflect',
                                                               horizontal_flip=True, vertical_flip=True)
    """
    # Previous settings

    image_gen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=80, zoom_range=0.4,
                                                                width_shift_range=0.4, height_shift_range=0.4,
                                                                fill_mode='reflect')
    mask_gen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=80, zoom_range=0.4,
                                                               width_shift_range=0.4, height_shift_range=0.4,
                                                               fill_mode='reflect')
    """
    print('Created ImageDataGenerator')

    # Fit the generator to the training dataset (90-10 validation split)
    image_gen.fit(x_train[:int(x_train.shape[0] * (1-valid_split))], augment=True, seed=seed)
    mask_gen.fit(y_train[:int(y_train.shape[0] * (1-valid_split))], augment=True, seed=seed)

    # Use the generators to create training data
    x = image_gen.flow(x_train[:int(x_train.shape[0] * (1-valid_split))], batch_size=batch_size, shuffle=True,
                       seed=seed)
    y = mask_gen.flow(y_train[:int(y_train.shape[0] * (1-valid_split))], batch_size=batch_size, shuffle=True, seed=seed)
    print('Training Data Fitted')

    # Creating the validation Image and Mask generator (no augmentation)
    image_gen_val = tf.keras.preprocessing.image.ImageDataGenerator()
    mask_gen_val = tf.keras.preprocessing.image.ImageDataGenerator()

    # Fit the generator to the validation dataset (90-10 validation split)
    image_gen_val.fit(x_train[int(x_train.shape[0] * (1-valid_split)):], augment=True, seed=seed)
    mask_gen_val.fit(y_train[int(y_train.shape[0] * (1-valid_split)):], augment=True, seed=seed)

    # Use the generators to create validation data
    x_val = image_gen_val.flow(x_train[int(x_train.shape[0] * (1-valid_split)):], batch_size=batch_size, shuffle=True,
                               seed=seed)
    y_val = mask_gen_val.flow(y_train[int(y_train.shape[0] * (1-valid_split)):], batch_size=batch_size, shuffle=True,
                              seed=seed)
    print('Validation Data Fitted')

    # Zip the generators for use in training
    train_generator = zip(x, y)
    val_generator = zip(x_val, y_val)

    return train_generator, val_generator
