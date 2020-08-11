# Author: Ryan Lee
# Create Date: Aug. 8th, 2020
# Purpose: Test the UNet Architecture for semantic segmentation and counting

import tensorflow as tf # Tensorflow including the Keras package within
import os # os to access and use directories
import numpy as np # numpy arrays to store data
import random # for debugging consistency or sample output
import matplotlib.pyplot as plt # output test plots

from tqdm import tqdm # progress bars on database extraction
from skimage.transform import resize # resize images
from skimage.io import imread, imshow # show images as windows

##
# Data Preprocessing
##

# Image dimensions (modify to fit microscopy)
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

# Dataset paths
TRAIN_DATA_PATH = 'train/'
TEST_DATA_PATH = 'test/'

# Seeds for consistent runs for debugging (optional)
# seed = 42
# random.seed = seed
# np.random.seed = seed

# Purpose: Access directories to extract, resize, and transform datasets into numpy arrays
# Parameters:
#   data_path: dataset directory address (string)
#   height: image height (int)
#   width: image width (int)
#   train: input directory is training set (with mask data) or not (bool)
def create_image_arrays(data_path, height, width, train):
    # Assign image ids through the directory
    image_ids = next(os.walk(data_path))[1]

    # Initialize numpy arrays to store both the color image and mask
    x_set = np.zeros((len(image_ids), height, width, IMG_CHANNELS), dtype=np.uint8)

    if train:
        y_set = np.zeros((len(image_ids), height, width, 1), dtype=np.bool)

    # Loop through every image using given path and unique folder identifier
    for i, id_ in tqdm(enumerate(image_ids), total=len(image_ids)):
        path = data_path + id_
        img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
        img = resize(img, (height, width), mode='constant', preserve_range=True)
        x_set[i] = img

        if train:
            # A set of unique boolean mask images for each object instance
            mask = np.zeros((height, width, 1), dtype=np.bool)
            for mask_file in next(os.walk(path + '/masks/'))[2]:
                mask_ = imread(path + '/masks/' + mask_file)
                mask_ = np.expand_dims(resize(mask_, (height, width), mode='constant',
                                              preserve_range=True), axis=-1)
                mask = np.maximum(mask, mask_)
            y_set[i] = mask

    if train:
        return x_set, y_set

    return x_set

# Use create_image_arrays() to turn the dataset into arrays
x_train, y_train = create_image_arrays(TRAIN_DATA_PATH, IMG_HEIGHT, IMG_WIDTH, True)
x_test = create_image_arrays(TEST_DATA_PATH, IMG_HEIGHT, IMG_WIDTH, False)

# Purpose: U-Net Architecture with 'same' padding modification and elu activation
# Parameters:
#   height: image height (int)
#   width: image width (int)
def unet_model(height, width):
    # Encoder: Input with normalization into [0,1] for a color (3-channel) image with specified width and height
    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    norm = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    # Five cycles of Convolution -> Dropout -> Convolution -> Max Pooling with ELU activations
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(norm)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(c5)

    # Decoder: Sequence with Transpose Convolutions back to the original dimensions
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(c9)

    # Final 1x1 Convolutions
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    # Set model with Adam Optimizer
    u_model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    u_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    u_model.summary()

    return u_model

# Create the UNet Architecture model
model = unet_model(IMG_HEIGHT, IMG_WIDTH)

# Checkpoints to keep the best weights
checkpoint_path = "checkpoints/test.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoints/callbacks to stop and save before overfitting
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    cp_callback
]

# Model training with a 10% validation split
results = model.fit(x_train, y_train, validation_split=0.1, batch_size=16, epochs=20,
                    callbacks=callbacks)

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
imshow(x_test[idx])
plt.show()
