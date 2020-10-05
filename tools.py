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
from scipy import ndimage  # Part of watershed calculation to find markers
from scipy import spatial  # KD Tree used to locate the nearest sperm in the label
from math import sqrt, pow  # Math functions to manually calculate the distances if there is only one label

# Constant for thresholding sperm counting
predict_threshold = 0.94
min_distance = 4


# Purpose: Basic mask model prediction output
# Parameters:
#   x_test: testing dataset
#   model: CNN model used
def pred_show(x_test, model):
    # Have a random or chosen predicted image
    index_type = input('Choose index type (random, chosen)')
    if index_type == 'random':
        idx = random.randint(0, len(x_test))
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
#   x_test: testing dataset
#   model: CNN model used
#   idx: index of test data to be segmented
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
    print("[COUNT] {} unique instances found".format(len(np.unique(labels)) - 1))

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
        cv2.circle(x_img_cpy, (int(x), int(y)), int(r), (0, 255, 0), 1)
        cv2.putText(x_img_cpy, "{}".format(label), (int(x) - 10, int(y) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    (0, 0, 255), 1)

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


# Purpose: Use watershed to predict the number of sperm with coordinates
# Parameters:
#   image: label to count
def count(image, min_dist=min_distance):
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

        # Draw a circle and text enclosing the detected region
        ((x, y), r) = cv2.minEnclosingCircle(c)
        label_xy.append((x, y))

    return sperm_count, label_xy


# Purpose: Traverse the directory to match labels with the predicted ground truths at 20x
def metrics(data_path, label_predict, label_truth, resized_height, resized_width, height, width, distance_threshold,
            scale):
    # Declare true and false positives
    tp = 0
    fp = 0
    fn = 0

    # Calculate # of cut-outs from the original image
    height_ratio = int(1024 // resized_height)
    width_ratio = int(1024 // resized_width)

    # Get list of predicted images in a directory
    predict_list = os.listdir(label_predict)

    # Loop through every image using given path and unique folder identifier
    for idx in tqdm(range(len(predict_list))):
        if scale == 'single':
            idx = random.randint(0, len(predict_list))
            print('Index: ' + str(idx))
        image = predict_list[idx]
        path = label_predict + image
        img = imread(path)
        
        ground_truth = list()
        predict_num = int(image[-7])
        if height_ratio == 2:
            if predict_num == 0:
                series = [0, 1, 4, 5]
            elif predict_num == 1:
                series = [2, 3, 6, 7]
            elif predict_num == 2:
                series = [8, 9, 12, 13]
            elif predict_num == 3:
                series = [10, 11, 14, 15]
            else:
                print('Out of range')
        else:
            series = range(16)

        # Traversal by row left to right
        for i in range(height_ratio):
            for j in range(width_ratio):
                img_path = label_truth + image[:-7] + str(series[i * height_ratio + j]) + image[-6:]
                truth = imread(img_path)
                ground_truth.append(truth)

        if scale == 'single':
            # Get a picture and convert it to 3 channels to draw on
            path = data_path + image
            pic = imread(path)
            pic = gray2rgb(pic)

            # Pass to count_metric to calculate the metrics for this one image
            precision, recall, f1, drawn = count_metric(height_ratio, width_ratio, height, width, img, ground_truth,
                                                        distance_threshold, scale, pic)

            # Show the image with circles drawn for tp, fp, fn
            imshow(drawn)
            plt.show()
            return precision, recall, f1
        else:
            # If full dataset, accumulate true positives, false positives, and false negatives for the final calculation
            tp_, fp_, fn_ = count_metric(height_ratio, width_ratio, height, width, img, ground_truth,
                                         distance_threshold, scale)
            tp += tp_
            fp += fp_
            fn += fn_

    # Calculate precision, recall, and F1 score
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1


# Purpose: Use the numpy arrays of the predicted label and ground truth to calculate precision, recall, and F1-score
# Parameters:
#   height_ratio: height ratio of slicing
#   width_ratio: width ratio of slicing
#   height: height of image
#   width: width of image
#   label: input predicted label
#   ground_truth: 3D array of the ground truth based on 20x magnification
#   distance_threshold: Distance for a sperm to be considered a true positive (must be within a sperm's radius)
def count_metric(height_ratio, width_ratio, height, width, label, ground_truth, distance_threshold, scale, pic=None):
    # Get the predicted coordinates from count() and create a truth_xy to append scaled coordinates to
    label_xy = count(label, distance_threshold)[1]
    truth_xy = list()

    # Declare true and false positives
    tp = 0
    fp = 0

    # Loop through every truth image and append the found coordinates
    for i in range(height_ratio):
        for j in range(width_ratio):
            idx = i * width_ratio + j
            coord_xy = count(ground_truth[idx], 3)[1]
            for coord in coord_xy:
                coord_x = coord[0] / width_ratio + j * (width / width_ratio)
                coord_y = coord[1] / height_ratio + i * (height / height_ratio)
                truth_xy.append((coord_x, coord_y))

    # Create a tree and then use it to find the nearest spatial coordinate until there are no more values left
    if len(truth_xy) > 1:
        truth_tree = spatial.KDTree(truth_xy)

    # Continue as long as there is a predicted sperm coordinate left
    while label_xy:
        predicted_xy = label_xy.pop()

        # Use the KD spatial tree if there is more tha none node
        if len(truth_xy) > 1:
            nearest = truth_tree.query(predicted_xy)
        # Otherwise, use use manual calculations
        elif len(truth_xy) == 1:
            nearest = [0, 0]
            nearest[0] = sqrt(pow(predicted_xy[0] - truth_xy[0][0], 2) + pow(predicted_xy[1] - truth_xy[0][1], 2))
            nearest[1] = 0

        # Only accept values within a distance threshold, then draw it on the image if only checking a single image
        if len(truth_xy) > 0 and nearest[0] < distance_threshold:
            if scale == 'single':
                cv2.circle(pic, (int(predicted_xy[0]), int(predicted_xy[1])), distance_threshold - 1, (0, 255, 0), 1)
            # Remove the found positive label
            del truth_xy[int(nearest[1])]
            # Re-generate a KD tree
            if len(truth_xy) > 1:
                truth_tree = spatial.KDTree(truth_xy)
            tp += 1
        # Otherwise, it's a false positive if there are no nearby true coordinates
        else:
            if scale == 'single':
                cv2.circle(pic, (int(predicted_xy[0]), int(predicted_xy[1])), distance_threshold - 1, (0, 0, 255), 1)
            fp += 1

    # Number of false negatives is the number of coordinates left
    fn = len(truth_xy)
    if scale == 'single':
        for false_n in truth_xy:
            cv2.circle(pic, (int(false_n[0]), int(false_n[1])), distance_threshold - 1, (255, 0, 0), 1)

    # Calculate precision, recall, and F1 score
    if tp != 0:
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


# Purpose: Evaluate the model using the model.evaluate function
# Parameters:
#   model: trained model used to predict images
#   data_from: data path to obtain images and names from
#   predict_to: data path to store predicted labels
def predict_set(model, data_from, predict_to, threshold=predict_threshold):
    # Create an iterable list through the directory
    imagelist = os.listdir(data_from)

    # Loop through every image using given path and unique folder identifier
    for image in tqdm(imagelist):
        # Get image from the name
        path = data_from + image
        img = imread(path)

        # Convert into three channels and expand dims to input to model
        img_rgb = gray2rgb(img)
        img_in = np.expand_dims(img_rgb, axis=0)

        # Convert model prediction to a binary label using a threshold after predicting
        predict = model.predict(img_in, verbose=0)
        predict_thresh = img_as_ubyte((predict > threshold).astype(np.bool))

        # Reformat the label to the correct dimensions
        predict_img = np.squeeze(predict_thresh)

        # Save the predicted label
        imsave(predict_to + image, predict_img, check_contrast=False)


def metrics_optimize(model, data_from, predict_path, resized_height, resized_width, height, width):
    # Lists of metric outputs
    precisions = list()
    recalls = list()
    f1s = list()

    # Input range of parameters to be tested
    predict_thresh_str = input('Input list of prediction thresholds (separated by a space): ')
    min_dist_str = input('Input list of minimum distances (separated by a space): ')
    predict_thresh_list = predict_thresh_str.split()
    min_dist_list = min_dist_str.split()

    for predict_thresh in predict_thresh_list:
        predict_set(model, data_from, predict_path, float(predict_thresh))
        for min_dist in min_dist_list:
            print('Prediction threshold:' + str(predict_thresh) + ' / Minimum distance: ' + str(min_dist))
            precision, recall, f1 = metrics(data_from, predict_path, 'Predict_20x/', resized_height, resized_width,
                                            height, width, int(min_dist), 'full')
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            print('Precision: ' + str(precision) + ' / Recall: ' + str(recall) + ' / F1-score: ' + str(f1))

    for predict_thresh in predict_thresh_list:
        for min_dist in min_dist_list:
            print('Prediction threshold:' + str(predict_thresh) + ' / Minimum distance: ' + str(min_dist))
            print('Precision: ' + str(precisions.pop(0)) + ' / Recall: ' + str(recalls.pop(0)) + ' / F1-score: '
                  + str(f1s.pop(0)))
