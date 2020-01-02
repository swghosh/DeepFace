#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras

IMAGE_SIZE = (152, 152)
CHANNELS = 3
NUM_CLASSES = 8631

BATCH_SIZE = 8
LEARN_RATE = 0.01 * (BATCH_SIZE / 128)
MOMENTUM = 0.9
EPOCHS = 15

CL_PATH = '/path/to/train_images_v1.txt'
# Note that the folder 'tfrecords' contains two sub folders :train, test
# train (contains training tfrecords), test (contains testing tfrecords)
DATASET_PATH = '/path/to/tfrecords'  
TB_PATH = '/path/to/tensorboard_logs'

keras.backend.clear_session()

from ..deepface import dataset
train, val = dataset.get_train_test_dataset(CL_PATH, DATASET_PATH, IMAGE_SIZE, BATCH_SIZE)
# these are essential values that have to be set
# in order to determine the right number of steps per epoch
train_samples, val_samples = 2307424, 25893
# this value is set so as to ensure
#  proper shuffling of dataset
dataset.SHUFFLE_BUFFER = train_samples
print('train.num_classes == ', train.num_classes)
print('validate.num_classes == ', val.num_classes)
#assert train.num_classes == val.num_classes == NUM_CLASSES

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
    patience=1, min_lr=0.0001, verbose=1) # mandatory step in training, as specified in paper
tensorboard = keras.callbacks.TensorBoard(TB_PATH)
checkpoints = keras.callbacks.ModelCheckpoint('weights.{epoch:02d}_{val_acc:.4f}.hdf5',
    monitor='val_acc', save_weights_only=True)

cbs = [reduce_lr, checkpoints, tensorboard]

from ..deepface import deepface

model=deepface.create_deepface(IMAGE_SIZE, CHANNELS, NUM_CLASSES, LEARN_RATE, MOMENTUM) 

model.fit(train.data, steps_per_epoch=train_samples // BATCH_SIZE + 1,
    validation_data=val.data, validation_steps=val_samples // BATCH_SIZE + 1,
    callbacks=cbs, epochs=EPOCHS)
model.save('model.h5')
