import cv2
import imutils
import matplotlib.pyplot as plt  # output test plot
import numpy as np
import random
import os

from tqdm import tqdm  # progress bars on database extraction
from skimage import img_as_ubyte
from skimage.io import imshow, imread, imsave  # show images as windows
from skimage.transform import resize  # resize images
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage


# Purpose: Basic mask model prediction output
# Parameters:
#   x_test: testing dataset
#   model: CNN model used
def pred_show(x_test, model):
    # Predict random example microscopy image from test set
    index_type = input('Choose index type (random, chosen)')
    if index_type == 'random':
        idx = random.randint(0, len(x_test))
    elif index_type == 'chosen':
        idx = int(input('Select index:'))
    x = np.array(x_test[idx])
    x = np.expand_dims(x, axis=0)
    predict = model.predict(x, verbose=1)
    print('Pre-conversion')
    print(np.shape(predict))
    print(predict[0, :, :, 0])

    # Current prediction set to be above 50% confidence
    print('Post-conversion')
    predict = (predict > 0.1).astype(np.uint8)
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
def watershed_pred(x_test, model, idx):
    x_img = np.array(x_test[idx])
    x_img_cpy = np.copy(x_img)
    x_img_exp = np.expand_dims(x_img, axis=0)
    predict = model.predict(x_img_exp, verbose=1)

    # Current prediction set to be above 50% confidence
    predict = (predict > 0.1).astype(np.uint8)

    # Create numpy image to be used in watershed
    image = np.squeeze(predict[0])

    # Euclidian distance from background using distance_transform
    dist = ndimage.distance_transform_edt(image)
    local_max = peak_local_max(dist, indices=False, min_distance=6, labels=image)

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
    plt.figure(1)
    imshow(x_img)
    plt.figure(2)
    imshow((image*255/image.max()).astype(np.uint8))
    plt.figure(3)
    imshow((dist*255/dist.max()).astype(np.uint8))
    plt.figure(4)
    imshow(x_img_cpy)
    plt.show()


# Purpose: Slice up a directory of larger images and masks into a desired size, currently ignores the bottom and right
# edges if there is a remainder
# Parameters:
#   tiff_path: original image path
#   label_path: original mask path
#   data_path: cut-up image path
#   mask_path: cut-up mask path
#   height: original height
#   width: original width
#   height_final: desired image height
#   width_final: desired image width
def slice_data(tiff_path, label_path, data_path, mask_path, height, width, height_final, width_final):
    # Assign image ids through the directory
    imagelist = os.listdir(tiff_path)

    # Calculate # of cut-outs from the original image
    height_ratio = height // height_final
    width_ratio = width // width_final

    # Loop through every image using given path and unique folder identifier
    for image in tqdm(imagelist):
        path = tiff_path + image
        img = imread(path)
        img = np.expand_dims(resize(img, (height, width), mode='constant', preserve_range=True), axis=-1)

        for i in range(height_ratio):
            for j in range(width_ratio):
                sliced_img = img[i * height_final:(i + 1) * height_final, j * width_final:(j + 1) * width_final]
                sliced_img = sliced_img.astype(np.uint8)
                imsave(data_path + image[:-6] + '_' + str(i * 4 + j) + image[-6:], sliced_img, check_contrast=False)

    # Assign image ids through the directory
    masklist = os.listdir(label_path)

    # Loop through every image using given path and unique folder identifier
    for mask in tqdm(masklist):
        path = label_path + mask
        img = imread(path)
        # img = np.expand_dims(resize(img, (width, height), mode='constant', preserve_range=True), axis=-1)

        for i in range(height_ratio):
            for j in range(width_ratio):
                sliced_img = img[i * height_final:(i + 1) * height_final, j * width_final:(j + 1) * width_final]
                sliced_img = (sliced_img.astype(bool) * 255).astype(np.uint8)
                imsave(mask_path + mask[:-8] + '_' + str(i * 4 + j) + mask[-8:], sliced_img, check_contrast=False)


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


def blank_filter(image_from, mask_from, image_to, mask_to):
    # Assign image ids through the directory
    masklist = os.listdir(mask_from)

    # Loop through every image using given path and unique folder identifier
    for mask in tqdm(masklist):
        mask_path = mask_from + mask
        mask_img = imread(mask_path)

        if np.count_nonzero(mask_img) > 30:
            imsave(mask_to + mask, mask_img, check_contrast=False)

            image = mask[:-8] + 'BF.png'
            image_path = image_from + image
            raw_image = imread(image_path)
            imsave(image_to + image, raw_image, check_contrast=False)


# def filter
#  Change data generator to find its file names based on the label
#  Some instances of class 1, overwhelming number of class 0's --> unbalanced datasets
#  Can easily run into problems doing it automatically
#  Staining might not be full-size sometimes
#  Try removing pure background labels - can do automatically <--
#  Hopefully
#  https://www.youtube.com/c/sentdex/playlists
