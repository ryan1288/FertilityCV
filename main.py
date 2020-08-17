# Author: Ryan Lee
# Create Date: Aug. 8th, 2020
# Purpose: Test the UNet Architecture for semantic segmentation and counting

import tensorflow as tf  # Tensorflow including the Keras package within
import os  # os to access and use directories
import numpy as np  # numpy for array operations

from tensorflow.keras.models import load_model  # Load saved model

# User-defined functions
from AugmentedGenerator import create_test_arrays, create_train_arrays, create_generators  # data importer functions
from model import create_unet  # U-Net CNN model
from tools import pred_show, watershed_pred  # Test model prediction

# Image dimensions (modify to fit microscopy)
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
BATCH_SIZE = 16
EPOCHS = 20
VALID_SPLIT = 0.1

# Dataset paths
TRAIN_DATA_PATH = 'train/'
TEST_DATA_PATH = 'test/'

# Checkpoints to keep the best weights
checkpoint_path = "checkpoints/test.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoints/callbacks to stop and save before overfitting
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
]

# Initialize dataset variables
x_test = x_train = y_train = model = None

# Set GPU memory growth
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


if __name__ == '__main__':
    # Initialize state as non-exit
    state = 'start'

    # Prompt until program is terminated
    while state != 'exit':
        # Select starting state
        state = input('Select mode: (data, load_data, train, load_model, test, exit)')

        if state == 'data':
            # Use create_image_arrays() to turn the dataset into arrays
            x_train, y_train = create_train_arrays(TRAIN_DATA_PATH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
            x_test = create_test_arrays(TEST_DATA_PATH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
            np.save('numpy_data/x_train', x_train)
            np.save('numpy_data/y_train', y_train)
            np.save('numpy_data/x_test', x_test)

        elif state == 'load_data':
            # Load numpy arrays
            x_train = np.load('numpy_data/x_train.npy')
            y_train = np.load('numpy_data/y_train.npy')
            x_test = np.load('numpy_data/x_test.npy')

        elif state == 'train':
            # Use create_generators to make the generators for the model training
            train_generator, val_generator = create_generators(x_train, y_train, VALID_SPLIT, BATCH_SIZE)

            # Create the UNet Architecture model
            model = create_unet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

            # Using Image Generators with a 10% validation split
            results = model.fit(train_generator, validation_data=val_generator, validation_steps=10,
                                steps_per_epoch=250, epochs=EPOCHS, callbacks=callbacks, verbose=1)

            # Save model
            model.save("saved_models/model.h5")

        elif state == 'load_model':
            # Load model
            model = load_model('saved_models/model.h5')

        elif state == 'test':
            # Select test type
            test_type = input('Select test type: (basic, watershed)')

            # Check if data is loaded
            if x_test is None:
                print('Data not yet loaded')
            elif test_type == 'basic':
                pred_show(x_test, model)
            elif test_type == 'watershed':
                watershed_pred(x_test, model, 0)
