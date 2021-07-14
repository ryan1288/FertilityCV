# Author: Ryan Lee
# Creation Date: Aug. 8th, 2020
# Purpose: Test the UNet Architecture for semantic segmentation and counting

import tensorflow as tf  # Tensorflow including the Keras package within
import os  # os to access and use directories
import scipy.io as s_io  # Saving numpy arrays to matlab for graphing
import matplotlib.pyplot as plt  # For plotting diagrams

from tensorflow.keras.models import load_model  # Load saved model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard,  ModelCheckpoint  # Callbacks to save/evaluate

# User-defined functions
# Data importer functions
from datagen import create_generators
# U-Net Model
from model import create_unet, calculate_weight, weighted_binary_crossentropy, dice_coef, evaluate_model
# Test model prediction
from tools import watershed_pred, metrics, metrics_optimize, predict_set, plot_roc, predict_video, video_transitory
# Data manipulation tools
from data import check_data, blank_filter, preprocess, split_data, clean_data

# Constants
RAW_IMG_HEIGHT = 2424
RAW_IMG_WIDTH = 2424
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3
BATCH_SIZE = 8
EPOCHS = 10

# Dataset paths
IMAGE_PATH = 'E:/FertilityCV/'
FOLDER_NAME = 'Sample/'
DATA_SOURCE = IMAGE_PATH + FOLDER_NAME
TIFF_PATH = DATA_SOURCE + 'Export/'
SIZED_PATH = DATA_SOURCE + 'Original/'
FILTER_PATH = DATA_SOURCE + 'AutoFilter/'
DATA_PATH = DATA_SOURCE + 'Filtered/Data/'
LABEL_PATH = DATA_SOURCE + 'Filtered/Label/'
PREDICT_PATH = DATA_SOURCE + 'Predict/'
SEARCH_PATH = DATA_SOURCE + 'Search/'
ROC_PATH = DATA_SOURCE + 'ROC/'

VIDEO_PATH = 'videos/'
DATA_SAVE = 'numpy_data/'
MODEL_SAVE = 'saved_models/model'
SAVE_POSTFIX = '_testis'
MODEL_POSTFIX = SAVE_POSTFIX + '_' + FOLDER_NAME[0:-1]

# Checkpoints to keep the best weights
checkpoint_path = "checkpoints/test.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoints/callbacks to stop and save before overfitting
callbacks = [
    EarlyStopping(patience=2, monitor='val_loss'),
    TensorBoard(log_dir='./logs/' + MODEL_POSTFIX + '_cross', update_freq=1),
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
        state = input('Select mode: (tiff, filter, clean, split, weight, train, load_model, checkpoint, '
                      'evaluate, predict, metrics, metrics_optimize, test, roc, check, video, exit)')

        if state == 'tiff':
            # Splits up tiff files into data and label images
            preprocess(TIFF_PATH, SIZED_PATH + 'Data/', SIZED_PATH + 'Label/', IMG_HEIGHT, IMG_WIDTH)
            print('Preprocessing complete')

        elif state == 'filter':
            # Filters out empty labels and evident outliers (large white blobs covering > 25% of the label)
            blank_filter(SIZED_PATH + 'Data/', SIZED_PATH + 'Label/', FILTER_PATH + 'Data/', FILTER_PATH + 'Label/',
                         IMG_HEIGHT, IMG_WIDTH, DATA_SOURCE)
            print('Filtered blank images')

        elif state == 'clean':
            # Clean out labels without matching data images and vice versa
            clean_data(FILTER_PATH + 'Data/', FILTER_PATH + 'Label/')

        elif state == 'split':
            split_type = input('Training type: (normal, cross)')
            if split_type == 'normal':
                # Splits filtered dataset into training, validation, and test sets
                split_data(FILTER_PATH + 'Data/', FILTER_PATH + 'Label/', DATA_PATH, LABEL_PATH)
            elif split_type == 'cross':
                # Split into five folders for five-fold cross validation
                split_data(FILTER_PATH + 'Data/', FILTER_PATH + 'Label/', DATA_PATH, LABEL_PATH, True)

        elif state == 'weight':
            # Calculate the weight ratio of background/sperm for training
            calculate_weight(DATA_SOURCE + 'Filtered/Label/train/train/')

        elif state == 'train':
            train_type = input('Training type: (new, continue)')

            # Use create_generators to make the generators for the model training
            if not train_generator:
                train_generator, val_generator, test_generator = create_generators(DATA_PATH, LABEL_PATH, BATCH_SIZE)
                print('Generators created')

            # Create the UNet Architecture model
            if train_type == 'new':
                model = create_unet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
                print('CNN Model created')
            elif train_type == 'continue':
                old_name = input('Model to continue training: ')
                post_name = input('Name post-script after retraining: ')
                model = load_model('saved_models/' + old_name + '.h5',
                                   custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy,
                                                   'dice_coef': dice_coef})
                print('Model loaded')
            else:
                continue

            # Using Image Generators with a 10% validation split
            results = model.fit(train_generator, validation_data=val_generator,
                                steps_per_epoch=len(os.listdir(DATA_PATH + 'train/train/'))//BATCH_SIZE, epochs=EPOCHS,
                                validation_steps=len(os.listdir(DATA_PATH + 'valid/valid/'))//BATCH_SIZE,
                                callbacks=callbacks, verbose=1)
            print('Model trained')

            # Save model
            if train_type == 'new':
                model.save(MODEL_SAVE + MODEL_POSTFIX + '.h5')
            else:
                model.save('saved_models/' + old_name + '_' + post_name + '.h5')
            print('Model saved')

        elif state == 'load_model':
            model_type = input('Model Type: (default, manual)')
            if model_type == 'default':
                name = 'model' + MODEL_POSTFIX
            else:
                name = input('Model name:')
            # Load model
            model = load_model('saved_models/' + name + '.h5',
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
                overwrite_name = input('Model Name: ')
                # Overwrite model using better checkpoint
                model.save('saved_models/' + overwrite_name + '.h5')
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
            predict_set(model, DATA_PATH, PREDICT_PATH, test_only=True)

        elif state == 'metrics':
            # Determine scale of metric to calculate
            scale = input('Metric scale: (single, full)')
            # Calculates precision/recall based on a single image or the full dataset
            precision, recall, f1 = metrics(DATA_PATH, LABEL_PATH, PREDICT_PATH, scale)
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
            # Apply watershed algorithm and label predictions showing each step
            watershed_pred(DATA_PATH + 'train/train/', LABEL_PATH + 'train/train/', model, SEARCH_PATH)

        elif state == 'roc':
            if model is None:
                # Load model
                model = load_model(MODEL_SAVE + MODEL_POSTFIX + '.h5',
                                   custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy,
                                                   'dice_coef': dice_coef})
            # Set model name here
            model_name = input('Model name:')
            model = load_model('saved_models/' + model_name + '.h5',
                               custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy,
                                               'dice_coef': dice_coef})
            # Plot ROC Curve along with AUC (Area under curve)
            fpr1, tpr1, roc_auc1 = plot_roc(model, DATA_PATH, LABEL_PATH, ROC_PATH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

            # Plot ROC with ROC value in the legend
            fig, ax = plt.subplots(1, 1)
            ax.plot(fpr1, tpr1, label='Normal (area = %0.3f)' % roc_auc1)
            ax.plot([0, 1], [0, 1], 'r--')
            """
            # Set second model (if necessary here)
            model = load_model('saved_models/model_normal_notail' + '.h5',
                               custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy,
                                               'dice_coef': dice_coef})
            fpr2, tpr2, roc_auc2 = plot_roc(model, DATA_PATH, LABEL_PATH, ROC_PATH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
            ax.plot(fpr2, tpr2, label='With Edges (area = %0.3f)' % roc_auc2)
            ax.plot([0, 1], [0, 1], 'b--')
            """
            # Set ROC plot limit before plotting
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic - Testis Cells')
            ax.legend(loc="lower right")
            plt.show()

            # Values are saved to be plotted in matlab if needed
            mat_dic = {'fpr1': fpr1, 'tpr1': tpr1, 'roc_auc1': roc_auc1}
            # , 'fpr2': fpr2, 'tpr2': tpr2, 'roc_auc2': roc_auc2}
            s_io.savemat("to_graph.mat", mat_dic)

        elif state == 'check':
            # Check input data to visualize as images and values
            check_data(DATA_PATH + 'train/train/', LABEL_PATH + 'train/train/')

        elif state == 'video':
            if model is None:
                # Load model
                model = load_model(MODEL_SAVE + MODEL_POSTFIX + '.h5',
                                   custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy,
                                                   'dice_coef': dice_coef})
            # Select test and data type
            video_name = input('Enter video file name:')
            edit_type = input('Edit type: (instant, slow)')
            if edit_type == 'instant':
                # Turn a video into frames and process each individual frame
                predict_video(VIDEO_PATH, model, video_name)
            else:
                video_transitory(VIDEO_PATH, model, video_name)
