# Author: Ryan Lee
# Create Date: Aug. 8th, 2020
# Purpose: Test the UNet Architecture for semantic segmentation and counting

import tensorflow as tf  # Tensorflow including the Keras package within
import os  # os to access and use directories
import numpy as np  # numpy for array operations

from tensorflow.keras.models import load_model  # Load saved model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint  # Callbacks to save/evaluate

# User-defined functions
from datagen import create_test_arrays, create_train_arrays, create_generators, create_ids  # data importer functions
from model import create_unet  # U-Net CNN model
from tools import pred_show, watershed_pred, slice_data, check_data, blank_filter  # Test model prediction
from generator import DataGenerator  # Sequence data generator class

# Image dimensions (modify to fit microscopy)
PRE_IMG_HEIGHT = 1040
PRE_IMG_WIDTH = 1392
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3
BATCH_SIZE = 4
EPOCHS = 10
VALID_SPLIT = 0.1

# Dataset paths
DATA_RAW_PATH = 'Data/'
LABEL_RAW_PATH = 'Label/'
DATA_PATH = 'Data_Filtered/'
LABEL_PATH = 'Label_Filtered/'
DATA_SAVE = 'numpy_data/'
MODEL_SAVE = 'saved_models/model'
SAVE_POSTFIX = '_filtered'

TEST_DATA_PATH = 'Data/'

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
        state = input('Select mode: (filter, slice, data, load_data, train, load_model, test, check, exit)')

        if state == 'filter':
            blank_filter(DATA_RAW_PATH, LABEL_RAW_PATH, DATA_PATH, LABEL_PATH)
            print('Filtered blank images')

        elif state == 'slice':
            # Cuts up image into desired final dimensions
            slice_data(DATA_RAW_PATH, LABEL_RAW_PATH, DATA_PATH, LABEL_PATH, PRE_IMG_HEIGHT, PRE_IMG_WIDTH,
                       IMG_HEIGHT, IMG_WIDTH)
            print('Sliced images')

        elif state == 'data':
            # Use create_image_arrays() to turn the dataset into arrays
            x_train, y_train = create_train_arrays(DATA_PATH, LABEL_PATH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
            # x_test = create_test_arrays(TEST_DATA_PATH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
            # train_ids, valid_ids = create_ids(TRAIN_DATA_PATH, VALID_SPLIT)
            np.save(DATA_SAVE + 'x_train' + SAVE_POSTFIX, x_train)
            np.save(DATA_SAVE + 'y_train' + SAVE_POSTFIX, y_train)
            # np.save(DATA_SAVE + 'x_test' + SAVE_POSTFIX, x_test)
            print('Numpy arrays saved')

        elif state == 'load_data':
            # Load numpy arrays
            x_train = np.load(DATA_SAVE + 'x_train' + SAVE_POSTFIX + '.npy')
            y_train = np.load(DATA_SAVE + 'y_train' + SAVE_POSTFIX + '.npy')
            # x_test = np.load(DATA_SAVE + 'x_test' + SAVE_POSTFIX + '.npy')
            print(np.shape(x_train))
            print(np.shape(y_train))
            # print(np.shape(x_test))
            print('Data loaded')

        elif state == 'train':
            # Use create_generators to make the generators for the model training
            train_generator, val_generator = create_generators(x_train, y_train, VALID_SPLIT, BATCH_SIZE)
            # params = {'dim': (IMG_HEIGHT, IMG_WIDTH),
            #          'batch_size': BATCH_SIZE,
            #          'n_channels': IMG_CHANNELS,
            #          'shuffle': True}
            # train_generator = DataGenerator(train_ids, **params)
            # val_generator = DataGenerator(valid_ids, **params)
            print('Generators created')

            # Create the UNet Architecture model
            model = create_unet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
            print('CNN Model created')

            # Using Image Generators with a 10% validation split
            results = model.fit(train_generator, validation_data=val_generator, steps_per_epoch=4000, epochs=EPOCHS,
                                validation_steps=400, callbacks=callbacks, verbose=1)
            print('Model trained')

            # Save model
            model.save(MODEL_SAVE + SAVE_POSTFIX + '.h5')
            print('Model saved')

        elif state == 'load_model':
            # Load model
            model = load_model(MODEL_SAVE + SAVE_POSTFIX + '.h5')
            print('Model loaded')

        elif state == 'test':
            # Select test and data type
            test_type = input('Select test type: (basic, watershed)')
            data_type = input('Select data type: (train, test)')

            # Allocate data type
            if data_type == 'train':
                data_in = x_train
            elif data_type == 'test':
                data_in = x_test
            else:
                continue

            # Check if data is loaded, if so, use chosen test
            if x_train is None:
                print('Data not yet loaded')
            elif test_type == 'basic':
                pred_show(data_in, model)
            elif test_type == 'watershed':
                watershed_pred(data_in, model, 0)  # Currently set to random image

        elif state == 'check':
            check_data(x_train, y_train)
