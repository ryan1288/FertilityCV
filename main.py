# Author: Ryan Lee
# Create Date: Aug. 8th, 2020
# Purpose: Test the UNet Architecture for semantic segmentation and counting

import tensorflow as tf  # Tensorflow including the Keras package within
import os  # os to access and use directories
import numpy as np  # numpy for array operations

from tensorflow.keras.models import load_model  # Load saved model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint  # Callbacks to save/evaluate

# User-defined functions
from datagen import create_train_arrays, create_generators  # data importer functions
from model import create_unet, calculate_weight, weighted_binary_crossentropy, dice_coef, evaluate_model  # U-Net CNN
from tools import pred_show, watershed_pred, metrics  # Test model prediction
from data import slice_data, check_data, blank_filter  # Data manipulation tools

# Constants
RAW_IMG_HEIGHT = 1040
RAW_IMG_WIDTH = 1392
RESIZE_IMG_HEIGHT = 512
RESIZE_IMG_WIDTH = 512
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3
BATCH_SIZE = 4
EPOCHS = 5
VALID_SPLIT = 0.1
METRIC_DISTANCE = 5

# Dataset paths
DATA_RAW_PATH = 'Data_Full/'
LABEL_RAW_PATH = 'Label_Full/'
SIZED_DATA_PATH = 'Data_10x/'
SIZED_LABEL_PATH = 'Label_10x/'
DATA_PATH = 'Data_Filtered_10x/'
LABEL_PATH = 'Label_Filtered_10x/'
DATA_SAVE = 'numpy_data/'
MODEL_SAVE = 'saved_models/model'
SAVE_POSTFIX = '_10x'

# Checkpoints to keep the best weights
checkpoint_path = "checkpoints/test.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoints/callbacks to stop and save before overfitting
callbacks = [
    EarlyStopping(patience=2, monitor='val_loss'),
    TensorBoard(log_dir='./logs'),
    ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
]

# Initialize dataset variables
x_test = x_train = y_train = model = watershed_counts = None

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
        state = input('Select mode: (slice, filter, data, load_data, weight, generator, train, load_model, evaluate, '
                      'metrics, test, check, exit)')

        if state == 'slice':
            # Cuts up image into desired final dimensions
            slice_data(DATA_RAW_PATH, LABEL_RAW_PATH, SIZED_DATA_PATH, SIZED_LABEL_PATH, RESIZE_IMG_HEIGHT,
                       RESIZE_IMG_WIDTH, IMG_HEIGHT, IMG_WIDTH)
            print('Sliced images')

        elif state == 'filter':
            # Filters out empty labels and evident outliers (large white blobs covering > 25% of the label)
            blank_filter(SIZED_DATA_PATH, SIZED_LABEL_PATH, DATA_PATH, LABEL_PATH, RESIZE_IMG_HEIGHT, RESIZE_IMG_WIDTH,
                         IMG_HEIGHT, IMG_WIDTH)
            print('Filtered blank images')

        elif state == 'data':
            # Use create_image_arrays() to turn the dataset into arrays
            x_train, y_train = create_train_arrays(DATA_PATH, LABEL_PATH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
            np.save(DATA_SAVE + 'x_train' + SAVE_POSTFIX, x_train)
            np.save(DATA_SAVE + 'y_train' + SAVE_POSTFIX, y_train)
            print('Numpy arrays saved')

        elif state == 'load_data':
            # Load numpy arrays
            x_train = np.load(DATA_SAVE + 'x_train' + SAVE_POSTFIX + '.npy')
            y_train = np.load(DATA_SAVE + 'y_train' + SAVE_POSTFIX + '.npy')
            print(np.shape(x_train))
            print(np.shape(y_train))
            print('Data loaded')

        elif state == 'weight':
            # Calculate the weight ratio of background/sperm for training
            calculate_weight(y_train)

        elif state == 'generator':
            # Use create_generators to make the generators for the model training
            train_generator, val_generator = create_generators(x_train, y_train, VALID_SPLIT, BATCH_SIZE)
            print('Generators created')

        elif state == 'train':
            # Create the UNet Architecture model
            model = create_unet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
            print('CNN Model created')

            # Using Image Generators with a 10% validation split
            results = model.fit(train_generator, validation_data=val_generator, steps_per_epoch=2000, epochs=EPOCHS,
                                validation_steps=200, callbacks=callbacks, verbose=1)
            print('Model trained')

            # Save model
            model.save(MODEL_SAVE + SAVE_POSTFIX + '.h5')
            print('Model saved')

        elif state == 'load_model':
            # Load model
            model = load_model(MODEL_SAVE + SAVE_POSTFIX + '.h5',
                               custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy,
                                               'dice_coef': dice_coef})
            print('Model loaded')

        elif state == 'evaluate':
            # Evaluate model using validation data generator
            results = evaluate_model(model, val_generator, BATCH_SIZE)
            print('Completed Evaluation')
            print(results)

        elif state == 'metrics':
            # Calculates precision/recall based on a single image or the full dataset
            metrics(DATA_PATH, LABEL_PATH, 'Label/', RESIZE_IMG_HEIGHT, RESIZE_IMG_WIDTH, IMG_HEIGHT, IMG_WIDTH,
                    METRIC_DISTANCE)

        elif state == 'test':
            # Select test and data type
            test_type = input('Select test type: (basic, watershed)')

            # Check if data is loaded, if so, use chosen test
            if x_train is None:
                print('Data not yet loaded')
            elif test_type == 'basic':
                pred_show(x_train, model)
            elif test_type == 'watershed':
                watershed_pred(x_train, y_train, model)

        elif state == 'check':
            # Check input data to visualize as images and values
            check_data(x_train, y_train)
