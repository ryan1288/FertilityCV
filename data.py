import os
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2

from skimage.transform import resize  # resize images
from tqdm import tqdm  # progress bars on database extraction
from skimage.io import imshow, imread, imsave  # show images as windows
from skimage.filters import threshold_otsu  # threshold labels from TIFF files


# Purpose: Displays the input data as images and values
# Parameters:
#   x_train: numpy array of input data
#   y_train: numpy array of input masks
def check_data(x_train, y_train):
    # Generate a random index in the dataset
    idx = random.randint(0, len(x_train))

    # Show the random image pair of bright field and label
    plt.figure(1)
    imshow(x_train[idx])
    plt.figure(2)
    imshow(y_train[idx])
    plt.show()

    # Show sample data values from the sample image pair
    print(x_train[idx, :, :, 0])
    print(x_train[idx, :, :, 1])
    print(x_train[idx, :, :, 2])
    print(y_train[idx, :, :, 0])


# Purpose: Quick filter of all labels that do not include a single sperm or more than quarter of the label is white
# Parameters:
#   data_from: original image path
#   label_from: original mask path
#   data_to: filtered image path
#   label_to: filtered mask path
#   height: image height
#   width: image width
def blank_filter(data_from, label_from, data_to, label_to, height, width):
    # Assign image ids through the directory
    masklist = os.listdir(label_from)

    # Loop through every image using given path and unique folder identifier
    for mask in tqdm(masklist):
        mask_path = label_from + mask
        mask_img = imread(mask_path)
        image_path = data_from + mask
        image = imread(image_path)

        # Count of positive labelled pixels
        pos_count = np.count_nonzero(mask_img)
        avg_intensity = np.mean(image)

        # Must include at least one sperm in the label, a sperm is ~12x12 (20x), ~6x6 (10x), ~3x3 (5x)
        # Also filter out white images where more than quarter of the label is white
        if (25 < pos_count < (height * width) / 5) and (70 < avg_intensity < 230):
            imsave(label_to + mask, mask_img, check_contrast=False)
            # If the passes through filter, save the BF image as well
            imsave(data_to + mask, image, check_contrast=False)


# Purpose: Pre-process stacked TIFF files into a set of images and labels
# Parameters:
#   data_from: original data path
#   data_to: filtered data path
#   alive_to: alive channel images
#   dead_to: dead channel images
#   height: image height
#   width: image width
def preprocess(data_from, data_to, label_to, height, width):
    tiff_num = 0
    # Assign image ids through the directory
    imagelist = os.listdir(data_from)

    # Loop through every image using given path and unique folder identifier
    for image in tqdm(imagelist):
        # Read image as stacked TIFF file
        image_path = data_from + image
        img = imread(image_path, plugin="pil")  # as_gray=True)

        # Calculate height ratio for a rounded number
        height_ratio = img[0].shape[0] // height
        width_ratio = img[0].shape[1] // width

        # Get the proper resized dimensions for proportional cuts
        resize_h = height * height_ratio
        resize_w = width * width_ratio

        # Loop through each image within the TIFF stack
        for i in range(len(img)):
            full_img = np.array(img[i])

            # Resize to a multiple of the height ratio and model input dimensions
            full_img = resize(full_img, (resize_h, resize_w), mode='constant', preserve_range=True)

            if i % 2 != 0:
                _, full_img = cv2.threshold(full_img, 240, 255, cv2.THRESH_TOZERO)

            # Traversal by row left to right
            for j in range(height_ratio):
                for k in range(width_ratio):
                    # Cut out a sliced portion
                    sliced_img = full_img[j * height:(j + 1) * height, k * width:(k + 1) * width]

                    # Convert data images to uint8 and save
                    if i % 2 == 0:
                        sliced_img = (sliced_img / sliced_img.max() * 255).astype(np.uint8)
                        imsave(data_to + str(tiff_num) + '_' + str(int(i / 2)) + '_' +
                               str(j * height_ratio + k) + '.png', sliced_img, check_contrast=False)
                    else:
                        # Check for fully positive images as the Otsu threshold works with > 1 colours
                        if np.mean(sliced_img) == 255:
                            sliced_img.fill(255)
                        # Only use thresholding if there is more than one colour in the image and isn't only background
                        elif np.mean(sliced_img) != 0:
                            # Use Otsu threshold to create binary label
                            sliced_img = (sliced_img > threshold_otsu(sliced_img)) * 255

                        sliced_img = sliced_img.astype(np.uint8)

                        # Save as label based on image in the sequence
                        imsave(label_to + str(tiff_num) + '_' + str(int(i / 2)) + '_' + str(j * height_ratio + k) +
                               '.png', sliced_img, check_contrast=False)

        tiff_num += 1


# Purpose: Combined alive and dead labels into one set of labels
# Parameters:
#   alive_from: alive labels data path
#   dead_from: dead labels data path
#   label_to: combined labels data path
def combine_labels(alive_from, dead_from, label_to):
    # Create image lists from the directories
    alive_list = os.listdir(alive_from)

    for i in tqdm(range(len(alive_list))):
        # Obtain the image name (identical for alive and dead) to get the image path
        image = alive_list[i]
        alive_path = alive_from + image
        dead_path = dead_from + image

        # Read the png image
        alive_img = imread(alive_path)
        dead_img = imread(dead_path)

        # Combined and save, with overlaps truncated by np.uint8
        label_img = alive_img | dead_img
        imsave(label_to + image, label_img, check_contrast=False)


# Purpose: Combined alive and dead labels into one set of labels
# Parameters:
#   alive_from: alive labels data path
#   dead_from: dead labels data path
#   label_to: combined labels data path
def split_data(data_from, label_from, data_to, label_to):
    # Create image lists from the directories
    label_list = os.listdir(label_from)
    print('There are ' + str(len(label_list)) + ' pairs of images.')

    # Acquire the split sizes
    valid_size = int(input('Validation set size: '))
    test_size = int(input('Test set size: '))

    # Initialize training counts
    valid_count = 0
    test_count = 0

    to_randomize = list(range(len(label_list)))
    np.random.shuffle(to_randomize)

    for i in tqdm(range(len(label_list))):
        # Obtain the image name (identical for alive and dead) to get the image path
        image = label_list[to_randomize[i]]
        data_source = data_from + image
        label_source = label_from + image

        # Read the png image
        data_img = imread(data_source)
        label_img = imread(label_source)

        # Randomly distribute the images based on pre-shuffled values
        if valid_count < valid_size:
            folder = 'valid/valid/'
            valid_count += 1
        elif test_count < test_size:
            folder = 'test/test/'
            test_count += 1
        else:
            folder = 'train/train/'

        # Save both to the respective folder
        imsave(data_to + folder + image, data_img, check_contrast=False)
        imsave(label_to + folder + image, label_img, check_contrast=False)


# Purpose: Combined alive and dead labels into one set of labels
# Parameters:
#   data: auto-filtered data path
#   label: auto-filtered label path
def clean_data(data, label):
    # Create image lists from the directories
    data_list = os.listdir(data)
    label_list = os.listdir(label)

    for data_id in tqdm(data_list):
        if data_id not in label_list:
            os.remove(data + data_id)
    for label_id in tqdm(label_list):
        if label_id not in data_list:
            os.remove(label + label_id)
