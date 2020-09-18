import os
import numpy as np
import matplotlib.pyplot as plt
import random

from skimage.transform import resize  # resize images
from tqdm import tqdm  # progress bars on database extraction
from skimage.io import imshow, imread, imsave  # show images as windows


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
    # Assign image ids through the directory
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
    idx = random.randint(0, 100)
    plt.figure(1)
    imshow(x_train[idx])
    plt.figure(2)
    imshow(y_train[idx])
    plt.show()
    print(x_train[idx, :, :, 0])
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

        # Count of positive labelled pixels
        pos_count = np.count_nonzero(mask_img)

        # Must include at least one sperm in the label, a sperm is ~12x12 (normal), ~6x6 (10x), ~3x3 (5x)
        # Also filter out white images where more than quarter of the label is white
        if 5 < pos_count < (height * width) / 4:
            imsave(label_to + mask, mask_img, check_contrast=False)

            image = mask[:-8] + 'BF.png'
            image_path = data_from + image
            raw_image = imread(image_path)
            imsave(data_to + image, raw_image, check_contrast=False)
