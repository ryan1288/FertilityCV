import tensorflow as tf
import numpy as np
import tensorflow.python.keras.backend as k


from tensorflow.keras.layers import Input, Lambda, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, concatenate  # Model
from tensorflow.keras import Model  # Compile and show summary of model


# Purpose: Custom Keras loss to train with weighted sperm cell labels to have more importance
# Parameters:
#   y_true: ground truth label
#   y_pred: predicted label
#   weight: pre-calculated ratio of background to sperm labelled pixels
def weighted_binary_crossentropy(y_true, y_pred, weight=151.28400868921892):
    y_true = k.clip(y_true, k.epsilon(), 1-k.epsilon())
    y_pred = k.clip(y_pred, k.epsilon(), 1-k.epsilon())
    logloss = -(y_true * k.log(y_pred) * weight + (1 - y_true) * k.log(1 - y_pred))
    return k.mean(logloss, axis=-1)


# Purpose: Scale-invariant metric that uses overlaps similar to MeanIoU
# Parameters:
#   y_true: ground truth label
#   y_pred: predicted label
#   smooth: smooths out coefficient parameter in the denominator
def dice_coef(y_true, y_pred, smooth=1):
    intersection = k.sum(k.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (k.sum(k.square(y_true), -1) + k.sum(k.square(y_pred), -1) + smooth)


# Purpose: Calculate and output ratios of class types in all labels of a dataset
# Parameters:
#   y_train: training label dataset
def calculate_weight(y_train):
    # Preliminary step to calculating ratio of - to + labels
    positive_label_ratio = np.sum(y_train)/np.size(y_train)
    print('Ratio of + labels to full dataset: ' + str(positive_label_ratio))

    # Used to train the model
    negative_to_positive = (1 - positive_label_ratio)/positive_label_ratio
    print('Ratio of - labels to + labels: ' + str(negative_to_positive))


# Purpose: U-Net Architecture with 'same' padding modification and elu activation
# Parameters:
#   height: image height (int)
#   width: image width (int)
def create_unet(width, height, channels):
    # Encoder: Input with normalization into [0,1] for a color (3-channel) image with specified width and height
    inputs = Input((width, height, channels))
    norm = Lambda(lambda x: x / 127.5 - 1)(inputs)

    # Five cycles of Convolution -> Dropout -> Convolution -> Max Pooling with ELU activations
    c1 = Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(norm)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(c5)

    # Decoder: Sequence with Transpose Convolutions back to the original dimensions
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(c9)

    # Final 1x1 Convolutions
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    # Set model with Adam Optimizer
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss=weighted_binary_crossentropy, metrics=['accuracy', dice_coef])
    model.summary()

    return model


def evaluate_model(model, val_generator, batch_size):
    # Evaluate using model.evaluate using the entire generator set
    results = model.evaluate(val_generator, batch_size=batch_size)
    return results
