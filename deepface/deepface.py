#!/usr/bin/env python3
from tensorflow import keras
from os import path

IMAGE_SIZE = (152, 152)
CHANNELS = 3
NUM_CLASSES = 8631
LEARN_RATE = 0.01
MOMENTUM = 0.9

def create_deepface(image_size=IMAGE_SIZE, channels=CHANNELS, num_classes=NUM_CLASSES, learn_rate=LEARN_RATE, momentum=MOMENTUM):
    """
    Deep CNN architecture primarily for Face Recognition,
    Face Verification and Face Representation (feature extraction) purposes
    "DeepFace: Closing the Gap to Human-Level Performance in Face Verification"
    CNN architecture proposed by Taigman et. al. (CVPR 2014)
    """

    wt_init = keras.initializers.RandomNormal(mean=0, stddev=0.01)
    bias_init = keras.initializers.Constant(value=0.5)

    """
    Construct certain functions 
    for using some common parameters
    with network layers
    """
    def conv2d_layer(**args):
        return keras.layers.Conv2D(**args, 
            kernel_initializer=wt_init, 
            bias_initializer=bias_init,
            activation=keras.activations.relu)
    def lc2d_layer(**args):
        return keras.layers.LocallyConnected2D(**args, 
            kernel_initializer=wt_init, 
            bias_initializer=bias_init,
            activation=keras.activations.relu)
    def dense_layer(**args):
        return keras.layers.Dense(**args, 
            kernel_initializer=wt_init, 
            bias_initializer=bias_init)

    """
    Create the network using
    tf.keras.layers.Layer(s)
    """
    deepface = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=(*image_size, channels), name='I0'),
        conv2d_layer(filters=32, kernel_size=11, name='C1'),
        keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same',  name='M2'),
        conv2d_layer(filters=16, kernel_size=9, name='C3'),
        lc2d_layer(filters=16, kernel_size=9, name='L4'),
        lc2d_layer(filters=16, kernel_size=7, strides=2, name='L5'),
        lc2d_layer(filters=16, kernel_size=5, name='L6'),
        keras.layers.Flatten(name='F0'),
        dense_layer(units=4096, activation=keras.activations.relu, name='F7'),
        keras.layers.Dropout(rate=0.5, name='D0'),
        dense_layer(units=num_classes, activation=keras.activations.softmax, name='F8')
    ], name='DeepFace')
    deepface.summary()

    """
    A tf.keras.optimizers.SGD will
    be used for training,
    and compile the model
    """
    sgd_opt = keras.optimizers.SGD(lr=learn_rate, momentum=momentum)
    cce_loss = keras.losses.categorical_crossentropy

    deepface.compile(optimizer=sgd_opt, loss=cce_loss, metrics=['accuracy'])
    return deepface

DOWNLOAD_PATH = 'https://github.com/swghosh/DeepFace/releases/download/weights-vggface2-2d-aligned/VGGFace2_DeepFace_weights_val-0.9034.h5.zip'
MD5_HASH = '0b21fb70cd6901c96c19ac14c9ea8b89'

def get_weights():
    filename = 'deepface.zip'
    downloaded_file_path = keras.utils.get_file(filename, DOWNLOAD_PATH, 
        md5_hash=MD5_HASH, extract=True)
    downloaded_h5_file = path.join(path.dirname(downloaded_file_path), 
        path.basename(DOWNLOAD_PATH).rstrip('.zip'))
    return downloaded_h5_file
    