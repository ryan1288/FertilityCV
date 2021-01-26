# Author: Ryan Lee
# Create Date: Aug. 8th, 2020
# Purpose: Test the UNet Architecture for semantic segmentation and counting

import tensorflow as tf  # Tensorflow including the Keras package within
import os  # os to access and use directories
import numpy as np  # numpy for array operations
import matplotlib.pyplot as plt  # For plotting diagrams

from tensorflow.keras.models import load_model  # Load saved model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint  # Callbacks to save/evaluate

# User-defined functions
from datagen import create_train_arrays, create_generators  # Data importer functions
from model import create_unet, calculate_weight, weighted_binary_crossentropy, dice_coef, evaluate_model  # U-Net Model
from tools import pred_show, watershed_pred, metrics, metrics_optimize, predict_set, plot_roc  # Test model prediction
from data import check_data, blank_filter, preprocess, combine_labels, split_data  # Data manipulation tools

# Constants
MAGNIFICATION = 10
RAW_IMG_HEIGHT = 2424
RAW_IMG_WIDTH = 2424
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3
BATCH_SIZE = 2
EPOCHS = 10
VALID_SPLIT = 0.1
METRIC_DISTANCE = 4

# Dataset paths
IMAGE_PATH = 'D:/FertilityCV/'
DATA_SOURCE = IMAGE_PATH + 'Testis_2/'
TIFF_PATH = DATA_SOURCE + 'Export/'
SIZED_PATH = DATA_SOURCE + 'Original/'
FILTER_PATH = DATA_SOURCE + 'AutoFilter/'
DATA_PATH = DATA_SOURCE + 'Filtered/Data/'
LABEL_PATH = DATA_SOURCE + 'Filtered/Label/'
PREDICT_PATH = DATA_SOURCE + 'Predict/'
ROC_PATH = DATA_SOURCE + 'ROC/'

DATA_SAVE = 'numpy_data/'
MODEL_SAVE = 'saved_models/model'
SAVE_POSTFIX = '_new_' + str(MAGNIFICATION) + 'x'
MODEL_POSTFIX = SAVE_POSTFIX + '_high_drop'
# Best is _high_drop

# Checkpoints to keep the best weights
checkpoint_path = "checkpoints/test.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoints/callbacks to stop and save before overfitting
callbacks = [
    EarlyStopping(patience=1, monitor='val_loss'),
    TensorBoard(log_dir='./logs/' + MODEL_POSTFIX),
    ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
]

# Initialize dataset variables
x_test = x_train = y_train = model = watershed_counts = train_generator = val_generator = test_generator = None
precision = recall = f1 = None

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
        state = input('Select mode: (tiff, slice, filter, split, data, load_data, weight, train, load_model, '
                      'checkpoint, evaluate, predict, metrics, metrics_optimize, test, roc, check, exit)')

        if state == 'tiff':
            channels = int(input('# Channels: (2, 3)'))
            preprocess(TIFF_PATH, SIZED_PATH + 'Data/', SIZED_PATH + 'Alive/', SIZED_PATH + 'Dead/', IMG_HEIGHT,
                       IMG_WIDTH, channels)
            print('Preprocessing complete')

        elif state == 'combine':
            combine_labels(SIZED_PATH + 'Alive/', SIZED_PATH + 'Dead/', SIZED_PATH + 'Label/')
            print('Labels combined')

        elif state == 'filter':
            # Filters out empty labels and evident outliers (large white blobs covering > 25% of the label)
            blank_filter(SIZED_PATH + 'Data/', SIZED_PATH + 'Label/', FILTER_PATH + 'Data/', FILTER_PATH + 'Label/',
                         IMG_HEIGHT, IMG_WIDTH)
            print('Filtered blank images')

        elif state == 'split':
            # Splits filtered dataset into training, validation, and test sets
            split_data(FILTER_PATH + 'Data/', FILTER_PATH + 'Label/', DATA_PATH, LABEL_PATH)

        elif state == 'data':
            # Use create_image_arrays() to turn the dataset into arrays
            x_train, y_train = create_train_arrays(DATA_PATH + 'train/train/', LABEL_PATH + 'train/train/',
                                                   IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
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

        elif state == 'train':
            # Use create_generators to make the generators for the model training
            if not train_generator:
                train_generator, val_generator, test_generator = create_generators(DATA_PATH, LABEL_PATH, BATCH_SIZE)
                print('Generators created')

            # Create the UNet Architecture model
            model = create_unet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
            print('CNN Model created')

            # Using Image Generators with a 10% validation split
            results = model.fit(train_generator, validation_data=val_generator,
                                steps_per_epoch=len(os.listdir(DATA_PATH + 'train/train/'))//BATCH_SIZE, epochs=EPOCHS,
                                validation_steps=len(os.listdir(DATA_PATH + 'valid/valid/'))//BATCH_SIZE,
                                callbacks=callbacks, verbose=1)
            print('Model trained')

            # Save model
            model.save(MODEL_SAVE + MODEL_POSTFIX + '.h5')
            print('Model saved')

        elif state == 'load_model':
            # Load model
            model = load_model(MODEL_SAVE + MODEL_POSTFIX + '.h5',
                               custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy,
                                               'dice_coef': dice_coef})
            print('Model loaded')

        elif state == 'checkpoint':
            # Load model
            model = load_model('checkpoints/test.ckpt',
                               custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy,
                                               'dice_coef': dice_coef})
            print('Checkpoint model loaded')
            # Determine scale of metric to calculate
            overwrite = input('Overwrite? (y/n)')
            if overwrite == 'y':
                # Overwrite model using better checkpoint
                model.save(MODEL_SAVE + MODEL_POSTFIX + '.h5')
                print('Model overwritten')

        elif state == 'evaluate':
            # Use create_generators to make the generators for the model training
            if not train_generator:
                train_generator, val_generator, test_generator = create_generators(DATA_PATH, LABEL_PATH, BATCH_SIZE)
                print('Generators created')

            # Evaluate model using validation data generator
            results = evaluate_model(model, test_generator, 1, len(os.listdir(DATA_PATH + 'test/test/'))//BATCH_SIZE)
            print('Completed Evaluation')
            print(results)

        elif state == 'predict':
            # Predicts and outputs a set of labels into a directory
            predict_set(model, DATA_PATH, PREDICT_PATH)

        elif state == 'metrics':
            # Determine scale of metric to calculate
            scale = input('Metric scale: (single, full)')
            # Calculates precision/recall based on a single image or the full dataset
            precision, recall, f1 = metrics(DATA_PATH, LABEL_PATH, PREDICT_PATH, METRIC_DISTANCE, scale)
            print('Precision: ' + str(precision))
            print('Recall: ' + str(recall))
            print('F1-score: ' + str(f1))

        elif state == 'metrics_optimize':
            # Optimize a list of models if needed
            model_input = input('Input model names? (y, n) ')
            if model_input == 'y':
                models = input('Input list of model names (separated by a space): ')
                models = models.split()
            else:
                models = [MODEL_POSTFIX]

            # Input range of parameters to be tested
            predict_thresh_str = input('Input list of prediction thresholds (separated by a space): ')
            min_rad_str = input('Input list of minimum radii (separated by a space): ')
            predict_thresh_list = predict_thresh_str.split()
            min_rad_list = min_rad_str.split()

            # Outputs a range of thresholds and minimum distances for each model
            for model_name in models:
                model = load_model(MODEL_SAVE + model_name + '.h5',
                                   custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy,
                                                   'dice_coef': dice_coef})
                metrics_optimize(model, DATA_PATH, LABEL_PATH, PREDICT_PATH, model_name, predict_thresh_list,
                                 min_rad_list)

        elif state == 'test':
            if model is None:
                # Load model
                model = load_model(MODEL_SAVE + MODEL_POSTFIX + '.h5',
                                   custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy,
                                                   'dice_coef': dice_coef})
            # Select test and data type
            test_type = input('Select test type: (basic, watershed)')

            # Check if data is loaded, if so, use chosen test
            if x_train is None:
                print('Data not yet loaded')
            elif test_type == 'basic':
                pred_show(x_train, model)
            elif test_type == 'watershed':
                watershed_pred(x_train, y_train, model)

        elif state == 'roc':
            if model is None:
                # Load model
                model = load_model(MODEL_SAVE + MODEL_POSTFIX + '.h5',
                                   custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy,
                                                   'dice_coef': dice_coef})
            # Plot ROC Curve along with AUC (Area under curve)
            model = load_model(MODEL_SAVE + '_10x' + '.h5',
                               custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy,
                                               'dice_coef': dice_coef})
            fpr, tpr, roc_auc = plot_roc(model, DATA_PATH, LABEL_PATH, ROC_PATH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

            # Plot ROC with ROC value in the legend
            fig, ax = plt.subplots(1, 1)
            ax.plot(fpr, tpr, label='Default (area = %0.3f)' % roc_auc)
            ax.plot([0, 1], [0, 1], 'r--')

            model = load_model(MODEL_SAVE + SAVE_POSTFIX + '_high_drop' + '.h5',
                               custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy,
                                               'dice_coef': dice_coef})
            fpr, tpr, roc_auc = plot_roc(model, DATA_PATH, LABEL_PATH, ROC_PATH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

            ax.plot(fpr, tpr, label='High Dropout (area = %0.3f)' % roc_auc)
            ax.plot([0, 1], [0, 1], 'g--')

            model = load_model(MODEL_SAVE + '_slow_gen_10x_high_drop' + '.h5',
                               custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy,
                                               'dice_coef': dice_coef})
            fpr, tpr, roc_auc = plot_roc(model, DATA_PATH, LABEL_PATH, ROC_PATH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

            ax.plot(fpr, tpr, label='Low Learning Rate (area = %0.3f)' % roc_auc)
            ax.plot([0, 1], [0, 1], 'b--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic - Sperm-only')
            ax.legend(loc="lower right")
            plt.show()


        elif state == 'check':
            # Check input data to visualize as images and values
            check_data(x_train, y_train)
