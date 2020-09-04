import tensorflow as tf

from tensorflow.keras.layers import Input, Lambda, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras import Model


# Purpose: U-Net Architecture with 'same' padding modification and elu activation
# Parameters:
#   height: image height (int)
#   width: image width (int)
def create_unet(width, height, channels):
    # Encoder: Input with normalization into [0,1] for a color (3-channel) image with specified width and height
    inputs = Input((width, height, channels))
    norm = Lambda(lambda x: x / 127.5 - 1)(inputs)  # Change to [-1, 1]

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
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', MeanIoU(num_classes=2)])
    model.summary()
    # Can adjust additional parameters on the adam

    return model
