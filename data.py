import os
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import csv

from skimage.transform import resize  # resize images
from tqdm import tqdm  # progress bars on database extraction
from skimage.io import imshow, imread, imsave  # show images as windows
from skimage.filters import threshold_otsu  # threshold labels from TIFF files


# Purpose: Displays the input data as images and values
# Parameters:
#   x_train: numpy array of input data
#   y_train: numpy array of input masks
def check_data(data_src, label_src):
    # Get list of image names
    img_names = os.listdir(data_src)

    # Generate a random index in the dataset
    idx = random.randint(0, len(img_names))

    data_img = imread(data_src + img_names[idx])
    label_img = imread(label_src + img_names[idx])

    # Show the random image pair of bright field and label
    plt.figure(1)
    imshow(data_img)
    plt.figure(2)
    imshow(label_img)
    plt.show()

    # Show sample data values from the sample image pair
    print(data_img)
    print(label_img)


# Purpose: Quick filter of all labels that do not include a single sperm or more than quarter of the label is white or
#   if specific parts of each well needs to be removed
# Parameters:
#   data_from: original image path
#   label_from: original mask path
#   data_to: filtered image path
#   label_to: filtered mask path
#   height: image height
#   width: image width
#   source: folder that contains the top folder of the stored images
def blank_filter(data_from, label_from, data_to, label_to, height, width, source):
    # Assign image ids through the directory
    masklist = os.listdir(label_from)
    filter_type = input('Remove specific locations for each well? [y/n]')
    if filter_type == 'y':
        locations = list()
        with open(source + 'locations.csv', encoding="utf-8-sig") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                locations.append((row[0], row[1]))

    # Loop through every image using given path and unique folder identifier
    for mask in tqdm(masklist):
        mask_path = label_from + mask
        mask_img = imread(mask_path)
        image_path = data_from + mask
        image = imread(image_path)

        # Choose to filter by label contents or locations
        # Used to remove all edge images and blurry images near the wells
        # Otherwise, just cleans up images in AutoFilter to contain the same corresponding pairs
        if filter_type == 'y':
            # Only save images that is not within the location to be deleted
            if (mask.split('_')[1], mask.split('_')[2][:-4]) not in locations:
                imsave(label_to + mask, mask_img, check_contrast=False)
                # If the passes through filter, save the BF image as well
                imsave(data_to + mask, image, check_contrast=False)
        else:
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

    # Choose whether to split in pairs or singles
    tiff_type = input('Type of tiff file: (double, BF):')

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

            if i % 2 != 0 and tiff_type == 'double':
                _, full_img = cv2.threshold(full_img, 240, 255, cv2.THRESH_TOZERO)

            # Traversal by row left to right
            for j in range(height_ratio):
                for k in range(width_ratio):
                    # Cut out a sliced portion
                    sliced_img = full_img[j * height:(j + 1) * height, k * width:(k + 1) * width]

                    # Convert data images to uint8 and save, if a BF-only TIFF, only divided but none put in labels
                    if i % 2 == 0 or tiff_type == 'BF':
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


# Purpose: Split data into training, validation, and testing sets randomly
# Parameters:
#   data_from: input BF images directory
#   label_from: input labels directory
#   data_to: output BF images main directory
#   label_to: output labels main directory
#   five_fold: If 5-fold validation is necessary, splits it into 5 equal randomized sets of data to manually move
def split_data(data_from, label_from, data_to, label_to, five_fold=False):
    # Create image lists from the directories
    label_list = os.listdir(label_from)
    print('There are ' + str(len(label_list)) + ' pairs of images.')

    if not five_fold:
        # Acquire the split sizes
        valid_size = int(input('Validation set size: '))
        test_size = int(input('Test set size: '))

        # Initialize training counts
        valid_count = 0
        test_count = 0
    else:
        # 5-fold validation always uses 20% sets
        count = 0

    # Randomize all index values by shuffling to get a randomized list of the images
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

        if not five_fold:
            # Randomly distribute the images based on pre-shuffled values
            if valid_count < valid_size:
                folder = 'valid/valid/'
                valid_count += 1
            elif test_count < test_size:
                folder = 'test/test/'
                test_count += 1
            else:
                folder = 'train/train/'
        else:
            # Outputs 20% of the dataset into one folder each
            if count < len(label_list) / 5:
                folder = 'A/'
            elif count < len(label_list) * 2 / 5:
                folder = 'B/'
            elif count < len(label_list) * 3 / 5:
                folder = 'C/'
            elif count < len(label_list) * 4 / 5:
                folder = 'D/'
            else:
                folder = 'E/'
            count += 1

        # Save both to the respective folder
        imsave(data_to + folder + image, data_img, check_contrast=False)
        imsave(label_to + folder + image, label_img, check_contrast=False)


# Purpose: Clean up data by removing unwanted images or BF images that don't exist as a label and vice versa
# Parameters:
#   data: auto-filtered data path
#   label: auto-filtered label path
def clean_data(data, label):
    # Create image lists from the directories
    data_list = os.listdir(data)
    label_list = os.listdir(label)

    # One more check to clean up images that are in unwanted locations based on .csv file
    locations = list()
    with open('E:/FertilityCV/20x/locations.csv', encoding="utf-8-sig") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            locations.append((row[0], row[1]))
    for data_id in tqdm(data_list):
        # Only save images that is not within the location to be deleted
        if (data_id.split('_')[1], data_id.split('_')[2][:-4]) in locations:
            os.remove(data + data_id)

    # Remove data images that are not in the label dataset
    for data_id in tqdm(data_list):
        if data_id not in label_list:
            os.remove(data + data_id)
    # Remove label images that are not in the data dataset
    for label_id in tqdm(label_list):
        if label_id not in data_list:
            os.remove(label + label_id)
