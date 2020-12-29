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
    for image in tqdm(labels):
        path = label_path + image
        img = imread(path)
        img = np.expand_dims(resize(img, (height, width), mode='constant', preserve_range=True), axis=-1)
        y_set[i] = img

        path = image_path + image
        img = imread(path)
        img = np.expand_dims(resize(img, (height, width), mode='constant', preserve_range=True), axis=-1)
        x_set[i] = img
        i += 1

    return x_set, y_set


# Purpose: Create data augmented generators that include the train-validation split for model training
# Parameters:
#   x_train: input training data
#   y_train: output mask ground truth
#   batch_size: size of batch when training
def create_generators(x_train, y_train, batch_size):
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
    # Creating the validation Image and Mask generator (no augmentation)
    image_gen_val = tf.keras.preprocessing.image.ImageDataGenerator()
    mask_gen_val = tf.keras.preprocessing.image.ImageDataGenerator()

    # Creating the validation Image and Mask generator (no augmentation)
    image_gen_test = tf.keras.preprocessing.image.ImageDataGenerator()
    mask_gen_test = tf.keras.preprocessing.image.ImageDataGenerator()

    print('Created ImageDataGenerators')

    # Use the generators to create training data
    x = image_gen.flow_from_directory(x_train + 'train/', color_mode='rgb', batch_size=batch_size, class_mode=None,
                                      shuffle=True, seed=seed)
    y = mask_gen.flow_from_directory(y_train + 'train/', color_mode='grayscale', batch_size=batch_size, class_mode=None,
                                     shuffle=True, seed=seed)
    print('Training Generators Created')

    # Use the generators to create validation data
    x_val = image_gen_val.flow_from_directory(x_train + 'valid/', color_mode='rgb', batch_size=batch_size,
                                              class_mode=None, shuffle=True, seed=seed)
    y_val = mask_gen_val.flow_from_directory(y_train + 'valid/', color_mode='grayscale', batch_size=batch_size,
                                             class_mode=None, shuffle=True, seed=seed)
    print('Validation Generators Created')

    # Use the generators to create testing data
    x_test = image_gen_test.flow_from_directory(x_train + 'test/', color_mode='rgb', batch_size=batch_size,
                                                class_mode=None, shuffle=True, seed=seed)
    y_test = mask_gen_test.flow_from_directory(y_train + 'test/', color_mode='grayscale', batch_size=batch_size,
                                               class_mode=None, shuffle=True, seed=seed)
    print('Testing Generators Created')

    # Zip the generators for use in training
    train_generator = zip(x, y)
    val_generator = zip(x_val, y_val)
    test_generator = zip(x_test, y_test)

    return train_generator, val_generator, test_generator
