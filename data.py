import os
import numpy as np
import matplotlib.pyplot as plt
import random

from skimage.transform import resize  # resize images
from tqdm import tqdm  # progress bars on database extraction
from skimage.io import imshow, imread, imsave  # show images as windows
from skimage.filters import threshold_otsu  # threshold labels from TIFF files


# Purpose: Slice up a directory of larger images and masks into a desired size, currently ignores the bottom and right
# edges if there is a remainder
# Parameters:
#   data_from: original image path
#   label_from: original mask path
#   data_to: cut-up image path
#   label_to: cut-up mask path
#   height: original height
#   width: original width
#   height_final: desired image height
#   width_final: desired image width
def slice_data(data_from, label_from, data_to, label_to, height, width, height_final, width_final):
    # Create an iterable list through the directory
    imagelist = os.listdir(data_from)

    # Calculate # of cut-outs from the original image
    height_ratio = height // height_final
    width_ratio = width // width_final

    # Loop through every image using given path and unique folder identifier
    for image in tqdm(imagelist):
        path = data_from + image
        img = imread(path)
        img = np.expand_dims(resize(img, (height, width), mode='constant', preserve_range=True), axis=-1)

        # Traversal by row left to right
        for i in range(height_ratio):
            for j in range(width_ratio):
                sliced_img = img[i * height_final:(i + 1) * height_final, j * width_final:(j + 1) * width_final]
                sliced_img = sliced_img.astype(np.uint8)
                imsave(data_to + image[:-6] + '_' + str(i * height_ratio + j) + image[-6:], sliced_img,
                       check_contrast=False)

    # Assign image ids through the directory
    masklist = os.listdir(label_from)

    # Loop through every image using given path and unique folder identifier
    for mask in tqdm(masklist):
        path = label_from + mask
        img = imread(path)
        img = resize(img, (width, height), mode='constant', preserve_range=True)

        # Traversal by row left to right
        for i in range(height_ratio):
            for j in range(width_ratio):
                sliced_img = img[i * height_final:(i + 1) * height_final, j * width_final:(j + 1) * width_final]
                sliced_img = (sliced_img.astype(bool) * 255).astype(np.uint8)
                imsave(label_to + mask[:-8] + '_' + str(i * height_ratio + j) + mask[-8:], sliced_img,
                       check_contrast=False)


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
def blank_filter(data_from, label_from, data_to, label_to, resized_height, resized_width, height, width):
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
        if (int(resized_height / height * resized_width / width * 5) < pos_count < (height * width) / 5) and \
                (70 < avg_intensity < 230):
            imsave(label_to + mask, mask_img, check_contrast=False)
            # If the passes through filter, save the BF image as well
            imsave(data_to + mask, image, check_contrast=False)


# Purpose: Pre-process stacked TIFF files into a set of images and labels
# Parameters:
#   data_from: original data path
#   data_to: filtered data path
#   height: image height
#   width: image width
def preprocess(data_from, data_to, alive_to, dead_to, height, width):
    tiff_num = 0
    # Assign image ids through the directory
    imagelist = os.listdir(data_from)

    # Read one sample image for the dimensions
    image_path = data_from + imagelist[0]
    img = imread(image_path, plugin="tifffile")
    height_ratio = img[0].shape[0] // height
    width_ratio = img[0].shape[1] // width

    # Get the proper resized dimensions for proportional cuts
    resize_h = height * height_ratio
    resize_w = width * width_ratio

    # Loop through every image using given path and unique folder identifier
    for image in tqdm(imagelist):
        # Read image as stacked TIFF file
        image_path = data_from + image
        img = imread(image_path, plugin="tifffile")

        # Loop through each image within the TIFF stack
        for i in range(len(img)):
            full_img = np.array(img[i])

            # Resize to a multiple of the height ratio and model input dimensions
            full_img = resize(full_img, (resize_h, resize_w), mode='constant', preserve_range=True)

            # Traversal by row left to right
            for j in range(height_ratio):
                for k in range(width_ratio):
                    # Cut out a sliced portion
                    sliced_img = full_img[j * height:(j + 1) * height, k * width:(k + 1) * width]

                    # Convert data images to uint8 and save
                    if i % 3 == 0:
                        sliced_img = (sliced_img / sliced_img.max() * 255).astype(np.uint8)
                        imsave(data_to + str(tiff_num) + '_' + str(int(i / 3)) + '_' +
                               str(j * height_ratio + k) + '.png', sliced_img, check_contrast=False)
                    else:
                        # Check for fully positive images as the Otsu threshold works with > 1 colours
                        if np.mean(sliced_img) > 65000:
                            sliced_img.fill(255)
                        else:
                            # Use Otsu threshold to create binary label
                            sliced_img = (sliced_img > threshold_otsu(sliced_img)) * 255
                        sliced_img = sliced_img.astype(np.uint8)

                        # Save as alive or dead depending based on the sequence of stacked images
                        if i % 3 == 1:
                            imsave(alive_to + str(tiff_num) + '_' + str(int(i / 3)) + '_' +
                                   str(j * height_ratio + k) + '.png', sliced_img, check_contrast=False)
                        else:
                            imsave(dead_to + str(tiff_num) + '_' + str(int(i / 3)) + '_' +
                                   str(j * height_ratio + k) + '.png', sliced_img, check_contrast=False)

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
