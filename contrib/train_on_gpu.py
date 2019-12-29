#!/usr/bin/env python3

"""
We'll use tf.keras for
# training the Deep Face network
"""
import tensorflow as tf
from tensorflow import keras

"""
Certain constants are to 
be defined
"""
IMAGE_SIZE = (152, 152)
CHANNELS = 3
NUM_CLASSES = 8631

#TPU_WORKER = 'grpc://10.0.0.1:8470'

BATCH_SIZE = 1024
LEARN_RATE = 0.01 * (BATCH_SIZE / 128)
MOMENTUM = 0.9
EPOCHS = 15

CL_PATH = '/path_to/train_images_v1.txt'
DATASET_PATH = '/path_to/tfrecords'  # Note that the folder 'tfrecords' contains two sub folders :train, test
# train (contains training tfrecords), test (contains testing tfrecords)
TB_PATH = '/path_to/output/tensorboard_logs_folder'

"""
Initialise the TPU
and create the required
tf.distribute.Strategy
"""
keras.backend.clear_session()

#tpu_cluster = tf.contrib.cluster_resolver.TPUClusterResolver(tpu=TPU_WORKER)
#tf.contrib.distribute.initialize_tpu_system(tpu_cluster)
#strategy = tf.contrib.distribute.TPUStrategy(tpu_cluster)

"""
Prepare the data pipeline
for train, val images
"""
from deepface import dataset
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

"""
Add some tf.keras.callbacks.Callback(s)
to enhance op(s)
like TensorBoard visualisation
and ReduceLR
"""
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
    patience=1, min_lr=0.0001, verbose=1) # mandatory step in training, as specified in paper
tensorboard = keras.callbacks.TensorBoard(TB_PATH)
checkpoints = keras.callbacks.ModelCheckpoint('weights.{epoch:02d}_{val_acc:.4f}.hdf5',
    monitor='val_acc', save_weights_only=True)

cbs = [reduce_lr, checkpoints, tensorboard]

"""
Construct the DeepFace
network with TPU strategy
"""
from deepface import deepface
#with strategy.scope():
#    model = deepface.create_deepface(IMAGE_SIZE, CHANNELS, NUM_CLASSES, LEARN_RATE, MOMENTUM)

model=deepface.create_deepface(IMAGE_SIZE, CHANNELS, NUM_CLASSES, LEARN_RATE, MOMENTUM) 
"""
Train the model
"""
train_history = model.fit(train.data, steps_per_epoch=train_samples // BATCH_SIZE + 1,
    validation_data=val.data, validation_steps=val_samples // BATCH_SIZE + 1,
    callbacks=cbs, epochs=EPOCHS)
model.save('model.h5')

"""
Let's visualise how the
training went
"""
from matplotlib import pyplot as plt
def save_plots():
    """
    Save two plot(s) 
    1. Accuracy vs Epochs
    2. Loss vs Epochs
    """
    acc = train_history.history['acc']
    val_acc = train_history.history['val_acc']

    loss = train_history.history['loss']
    val_loss = train_history.history['val_loss']

    plt.figure(figsize=(8, 8))

    plt.subplot(2, 1, 1)
    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower right')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.xlabel('Epochs')
    plt.title('Loss')

    plt.savefig('epcoch_wise_loss_acc.png')

save_plots()
