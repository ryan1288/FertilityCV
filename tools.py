import cv2
import imutils
import matplotlib.pyplot as plt  # output test plot
import numpy as np
import random

from skimage.io import imshow  # show images as windows
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage


# Purpose: Basic mask model prediction output
# Parameters:
#   x_test: testing dataset
#   model: CNN model used
def pred_show(x_test, model):
    # Predict random example microscopy image from test set
    idx = random.randint(0, len(x_test))
    x = np.array(x_test[idx])
    x = np.expand_dims(x, axis=0)
    predict = model.predict(x, verbose=1)

    # Current prediction set to be above 50% confidence
    predict = (predict > 0.5).astype(np.uint8)

    # Show windows of predicted mask and image
    imshow(np.squeeze(predict[0]))
    plt.show()
    print(np.squeeze(predict[0]).shape)
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
    predict = (predict > 0.5).astype(np.uint8)

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
