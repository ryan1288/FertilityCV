import os
import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import random

from tqdm import tqdm  # progress bars on database extraction
from skimage import img_as_ubyte  # Convert to ubyte for saving
from skimage.color import gray2rgb  # Convert single
from skimage.io import imshow, imread, imsave  # Show, read, and save images
from skimage.feature import peak_local_max  # Use euclidian distances to find local maxes
from skimage.segmentation import watershed  # Watershed tool to find labels
from sklearn.metrics import roc_curve, auc  # roc curve tools
from scipy import ndimage  # Part of watershed calculation to find markers
from scipy import spatial  # KD Tree used to locate the nearest sperm in the label
from math import sqrt, pow  # Math functions to manually calculate the distances if there is only one label
from datagen import create_train_arrays  # To create the arrays for the ROC curve calculation

# Constant values for testing
predict_threshold = 0.92  # Thresholding sperm counting
min_distance = 6  # Minimum distance between local maxima
radius_threshold = 2  # Minimum radius of label to be considered a sperm

x10_min_dist = 6
x10_min_rad = 3
# 20x -
# 10x - 0.94 / 6 / 3


# Purpose: Basic mask model prediction output
# Parameters:
#   x_test: testing dataset
#   model: CNN model used
def pred_show(x_test, model):
    # Have a random or chosen predicted image
    index_type = input('Choose index type (random, chosen)')
    if index_type == 'random':
        idx = random.randint(0, len(x_test))
        print('Random index: ' + str(idx))
    elif index_type == 'chosen':
        idx = int(input('Select index: '))
    else:
        return

    # Obtain the image, and show the prediction values
    x = np.array(x_test[idx])
    x = np.expand_dims(x, axis=0)
    print(x.shape)
    print(x[:, :, :, 0])
    predict = model.predict(x, verbose=1)
    print('Pre-conversion')
    print(np.shape(predict))
    print(predict[0, :, :, 0])

    # Current prediction set to be above a set confidence
    print('Post-conversion')
    predict = (predict > predict_threshold).astype(np.uint8)
    print(np.shape(predict))
    print(predict[0, :, :, 0])

    # Show windows of predicted mask and image
    plt.figure(1)
    imshow(np.squeeze(predict[0]))
    plt.figure(2)
    imshow(x_test[idx])
    plt.show()


# Purpose: Create data augmented generators that include the train-validation split for model training
# Parameters:
#   x_test: testing dataset - image
#   y_test: testing dataset - label
#   model: CNN model used
def watershed_pred(x_test, y_test, model):
    # Have a random or chosen predicted image
    index_type = input('Choose index type (random, chosen)')
    if index_type == 'random':
        idx = random.randint(0, len(x_test))
        print('Random index: ' + str(idx))
    elif index_type == 'chosen':
        idx = int(input('Select index: '))
    else:
        return

    # Get the image from the data set
    x_img = np.array(x_test[idx])
    y_img = np.array(y_test[idx])
    x_img_cpy = np.copy(x_img)
    x_img_exp = np.expand_dims(x_img_cpy, axis=0)

    # Predict using the trained model
    predict = model.predict(x_img_exp, verbose=1)

    # Current prediction set to be above a set confidence
    predict = (predict > predict_threshold).astype(np.uint8)

    # Create numpy image to be used in watershed
    image = np.squeeze(predict[0])

    # Euclidian distance from background using distance_transform
    dist = ndimage.distance_transform_edt(image)
    local_max = peak_local_max(dist, indices=False, min_distance=min_distance, labels=image)

    # Connected component analysis before using watershed algorithm
    markers = ndimage.label(local_max, structure=np.ones((3, 3)))[0]
    labels = watershed(-dist, markers, mask=image)
    # Old labels that included every blob
    # print("[COUNT] {} unique instances found".format(len(np.unique(labels)) - 1))

    # Count of sperm
    sperm_count = 0

    # Loop through unique labels
    for label in np.unique(labels):
        # Ignore background regions
        if label == 0:
            continue

        # Label unique detected regions
        mask = np.zeros(image.shape, dtype="uint8")
        mask[labels == label] = 255

        # Grab largest contour in the mask
        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        c = max(contours, key=cv2.contourArea)

        # Draw a circle and text enclosing the detected region
        ((x, y), r) = cv2.minEnclosingCircle(c)
        if r > radius_threshold:
            cv2.circle(x_img_cpy, (int(x), int(y)), int(r), (0, 255, 0), 1)
            cv2.putText(x_img_cpy, "{}".format(sperm_count + 1), (int(x) - 10, int(y) - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, (0, 0, 255), 1)
            sperm_count += 1

    print("[COUNT] {} unique instances found".format(sperm_count))

    # Plot images for pipeline progression
    plt.figure(1)  # Original Image
    imshow(x_img)
    plt.figure(2)  # Predicted Label
    imshow((image*255/image.max()).astype(np.uint8))
    plt.figure(3)  # With Euclidian Distance
    imshow((dist*255/dist.max()).astype(np.uint8))
    plt.figure(4)  # Drawn Circles
    imshow(x_img_cpy)
    plt.figure(5)  # Original Label
    imshow(y_img)
    plt.show()


# Purpose: Counts the number of unique labels and returns the coordinates and total count
# Parameters:
#   image: label to count
#   min_dist: minimum distance between sperm centers threshold
#   rad_threshold: minimum radius of circle enclosing sperm threshold
def count(image, min_dist=min_distance, rad_threshold=radius_threshold):
    # Euclidian distance from background using distance_transform
    dist = ndimage.distance_transform_edt(image)
    local_max = peak_local_max(dist, indices=False, min_distance=min_dist, labels=image)

    # Connected component analysis before using watershed algorithm
    markers = ndimage.label(local_max, structure=np.ones((3, 3)))[0]
    labels = watershed(-dist, markers, mask=image)

    # Count number of sperm
    sperm_count = len(np.unique(labels)) - 1

    label_xy = []

    # Loop through unique labels
    for label in np.unique(labels):
        # Ignore background regions
        if label == 0:
            continue

        # Label unique detected regions
        mask = np.zeros(image.shape, dtype="uint8")
        mask[labels == label] = 255

        # Grab largest contour in the mask
        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        c = max(contours, key=cv2.contourArea)

        # Add the circle's label to the list
        ((x, y), r) = cv2.minEnclosingCircle(c)
        if r > rad_threshold:
            label_xy.append((x, y))

    return sperm_count, label_xy


# Purpose: Computes the recall, precision, and F1-Score metrics by traversing through folders
# Parameters:
#   data_path: images directory
#   label_path: label directory
#   predict_path: prediction directory
#   distance_threshold: minimum distance between sperm centers threshold
#   scale: single image or full dataset testing
#   rad_threshold: minimum radius of circle enclosing sperm threshold
def metrics(data_path, label_path, predict_path, distance_threshold, scale, rad_threshold=radius_threshold):
    # Early declaration of folders
    folder_list = ['test/test/']

    # Declare true and false positives
    tp = fp = fn = precision = recall = f1 = 0

    if scale == 'single':
        # Get random index in training dataset
        predict_list = os.listdir(data_path + folder_list[0])
        idx = random.randint(0, len(predict_list))
        print('Index: ' + str(idx))

        # Get a randomized idx if only a single image is selected
        image = predict_list[idx]
        print('Image: ' + image)

        # Get a picture and convert it to 3 channels to draw on
        path = data_path + folder_list[0] + image
        pic = imread(path)
        pic = gray2rgb(pic)

        # Get the label image to put through the model
        path = label_path + folder_list[0] + image
        img = imread(path)

        # Get the ground truth label to compare to the prediction
        path = predict_path + image
        ground_truth = imread(path)

        # Pass to count_metric to calculate the metrics for this one image
        precision, recall, f1, drawn = count_metric(img, ground_truth, distance_threshold, scale, rad_threshold, pic)

        # Show the image with circles drawn for tp, fp, fn
        imshow(drawn)
        plt.show()
        return precision, recall, f1

    elif scale == 'full':
        for folder in folder_list:
            predict_list = os.listdir(label_path + folder)
            # Loop through every image using given path and unique folder identifier
            for image in tqdm(predict_list):
                # Get the label image to put through the model
                path = predict_path + image
                img = imread(path)

                # Get the ground truth label to compare to the prediction
                path = label_path + folder + image
                ground_truth = imread(path)

                # If full dataset, accumulate true positives, false positives, and false negatives
                tp_, fp_, fn_ = count_metric(img, ground_truth, distance_threshold, scale, rad_threshold)
                tp += tp_
                fp += fp_
                fn += fn_

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1


# Purpose: Use the numpy arrays of the predicted label and ground truth to calculate precision, recall, and F1-score
# Parameters:
#   label: input predicted label
#   ground_truth: 3D array of the ground truth based on 20x magnification
#   distance_threshold: Distance for a sperm to be considered a true positive (must be within a sperm's radius)
#   scale: single image or full dataset testing
#   rad_threshold: minimum radius of circle enclosing sperm threshold
#   pic: input image to draw on for visualization (optional)
def count_metric(label, ground_truth, distance_threshold, scale, rad_threshold=radius_threshold, pic=None):
    # Declare true and false positives
    tp = fp = 0

    # Get the predicted coordinates from count() and create a label_xy to append scaled coordinates to
    label_xy = count(label, distance_threshold, rad_threshold)[1]

    # Count x_y coordinates in ground truth image
    coord_xy = count(ground_truth, x10_min_dist, x10_min_rad)[1]

    # Create a tree and then use it to find the nearest spatial coordinate until there are no more values left
    if len(coord_xy) > 1:
        truth_tree = spatial.KDTree(coord_xy)

    # Continue as long as there is a predicted sperm coordinate left
    while label_xy:
        predicted_xy = label_xy.pop()

        # Use the KD spatial tree if there is more tha none node
        if len(coord_xy) > 1:
            nearest = truth_tree.query(predicted_xy)
        # Otherwise, use use manual calculations
        elif len(coord_xy) == 1:
            nearest = [0, 0]
            nearest[0] = sqrt(pow(predicted_xy[0] - coord_xy[0][0], 2) + pow(predicted_xy[1] - coord_xy[0][1], 2))
            nearest[1] = 0

        # Only accept values within a distance threshold, then draw it on the image if only checking a single image
        if len(coord_xy) > 0 and nearest[0] < distance_threshold:
            if scale == 'single':
                cv2.circle(pic, (int(predicted_xy[0]), int(predicted_xy[1])), distance_threshold - 1, (0, 255, 0), 1)
            # Remove the found positive label
            del coord_xy[int(nearest[1])]
            # Re-generate a KD tree
            if len(coord_xy) > 1:
                truth_tree = spatial.KDTree(coord_xy)
            tp += 1
        # Otherwise, it's a false positive if there are no nearby true coordinates
        else:
            if scale == 'single':
                cv2.circle(pic, (int(predicted_xy[0]), int(predicted_xy[1])), distance_threshold - 1, (0, 0, 255), 1)
            fp += 1

    # Number of false negatives is the number of coordinates left
    fn = len(coord_xy)
    if scale == 'single':
        for false_n in coord_xy:
            cv2.circle(pic, (int(false_n[0]), int(false_n[1])), distance_threshold - 1, (255, 0, 0), 1)

    # Calculate precision, recall, and F1 score
    if tp + fp != 0:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
    # Case where there were no true positives, 0 scores
    else:
        precision = 0
        recall = 0
        f1 = 0

    # Return the drawn image and metrics if evaluating a single image, otherwise, provide values to accumulate
    if scale == 'single':
        return precision, recall, f1, pic
    else:
        return tp, fp, fn


# Purpose: Predict a set of images
# Parameters:
#   model: trained model used to predict images
#   data_from: data path to obtain images and names from
#   predict_to: data path to store predicted labels
#   threshold: binarization value between 0 and 1
#   test_only: all folders or only test folder
#   continuous: return continuous values pre-thresholding if enabled
def predict_set(model, data_from, predict_to, threshold=predict_threshold, test_only=False, continuous=False):
    # Create folder list for all three data types
    if test_only:
        folder_list = ['test/test/']
    else:
        folder_list = ['train/train', 'valid/valid', 'test/test/']

    # Loop through folder list
    for folder in folder_list:
        # Create an iterable list through the directory
        imagelist = os.listdir(data_from + folder)

        # For continuous output
        if continuous:
            predictions = np.zeros((len(imagelist), 256, 256), dtype=np.float64)
            i = 0

        # Loop through every image using given path and unique folder identifier
        for image in tqdm(imagelist):
            # Get image from the name
            path = data_from + folder + image
            img = imread(path)

            # Convert into three channels and expand dims to input to model
            img_rgb = gray2rgb(img)
            img_in = np.expand_dims(img_rgb, axis=0)

            # Convert model prediction to a binary label using a threshold after predicting
            predict = model.predict(img_in, verbose=0)

            # Option to output pre-thresholding for continuous output values
            if continuous:
                predictions[i] = np.squeeze(predict)
                print(predict)
                i = i + 1
            else:
                predict_thresh = img_as_ubyte((predict > threshold).astype(np.bool))

                # Reformat the label to the correct dimensions
                predict_img = np.squeeze(predict_thresh)

                # Save the predicted label
                imsave(predict_to + image, predict_img, check_contrast=False)

        # Return predictions for the single set if continuous
        if continuous:
            return predictions


# Purpose: Calculate metrics for multiple binarization and radius thresholds
# Parameters:
#   model: trained model used to predict images
#   data_from: data path to obtain images and names from
#   label_path: label path to obtain masks from
#   predict_path: data path of stored predicted labels
#   model_name: model name to be printed
#   predict_thresh_list: list of prediction binarization thresholds
#   min_rad_list: list of minimum radius thresholds
def metrics_optimize(model, data_path, label_path, predict_path, model_name, predict_thresh_list, min_rad_list):
    # Lists of metric outputs
    precisions = list()
    recalls = list()
    f1s = list()

    # Use metrics function on each prediction threshold to calculate individual metrics
    for predict_thresh in predict_thresh_list:
        predict_set(model, data_path, predict_path, float(predict_thresh))
        for min_rad in min_rad_list:
            print('Prediction threshold:' + str(predict_thresh) + ' / Minimum radius: ' + str(min_rad))
            precision, recall, f1 = metrics(data_path, label_path, predict_path, min_distance, 'full', int(min_rad))
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            print('Precision: ' + str(precision) + ' / Recall: ' + str(recall) + ' / F1-score: ' + str(f1))

    # Display a summary of all prediction results
    print('Model name: ' + model_name)
    for predict_thresh in predict_thresh_list:
        for min_rad in min_rad_list:
            print('Prediction threshold:' + str(predict_thresh) + ' / Minimum radius: ' + str(min_rad))
            print('Precision: ' + str(precisions.pop(0)) + ' / Recall: ' + str(recalls.pop(0)) + ' / F1-score: '
                  + str(f1s.pop(0)))


# Purpose: Plot ROC curve for a model
# Parameters:
#   model: trained model used to predict images
#   data_from: data path to obtain images and names from
#   label_path: label path to obtain masks from
#   roc_path: path to store generated roc curve
#   height: image height
#   width: image width
#   channels: # of image channels
def plot_roc(model, data_path, label_path, roc_path, height, width, channels):
    # Create and load test arrays
    x_test, y_test = create_train_arrays(data_path + 'test/test/', label_path + 'test/test/', height, width, channels)

    # Predicts and outputs a set of labels into a directory
    y_predict = predict_set(model, data_path, roc_path, test_only=True, continuous=True)

    # Unravel values to a 1D vectors to use roc_curve
    ground_truth_labels = y_test.ravel()
    score_value = y_predict.ravel()

    # Calculate false positive and true positive rates
    fpr, tpr, _ = roc_curve(ground_truth_labels, score_value)
    roc_auc = auc(fpr, tpr)

    # Plot ROC with ROC value in the legend
    fig, ax = plt.subplots(1, 1)
    ax.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic - Sperm-only - High Dropout')
    ax.legend(loc="lower right")
    plt.show()