#!/usr/bin/env python3

# We'll use tf.keras for
# training the Deep Face network
import tensorflow as tf
from tensorflow import keras

# Certain constants are to 
# be defined
IMAGE_SIZE = (152, 152)
CHANNELS = 3
NUM_CLASSES = 8631

TPU_WORKER = 'grpc://10.0.0.1:8470'

BATCH_SIZE = 1024
LEARN_RATE = 0.01 * (BATCH_SIZE / 128)
MOMENTUM = 0.9
EPOCHS = 15

CL_PATH = 'gs://bucket-name/VGGFace2_classlabels.txt'
DATASET_PATH = 'gs://bucket-name/VGGFace2_tfrecs'
TB_PATH = 'gs://bucket-name/vggface2_deepface_tensorboard'

# Initialise the TPU
# and create the required
# tf.distribute.Strategy
keras.backend.clear_session()

tpu_cluster = tf.contrib.cluster_resolver.TPUClusterResolver(tpu=TPU_WORKER)
tf.contrib.distribute.initialize_tpu_system(tpu_cluster)
strategy = tf.contrib.distribute.TPUStrategy(tpu_cluster)

# Create a function that will
# return a created network
# DeepFace
def create_deepface():
    """
    Construct certain functions 
    for using some common parameters
    with network layers
    """
    wt_init = keras.initializers.RandomNormal(mean=0, stddev=0.01)
    bias_init = keras.initializers.Constant(value=0.5)

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
        keras.layers.InputLayer(input_shape=(*IMAGE_SIZE, CHANNELS), name='I0'),
        conv2d_layer(filters=32, kernel_size=11, name='C1'),
        keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same',  name='M2'),
        conv2d_layer(filters=16, kernel_size=9, name='C3'),
        lc2d_layer(filters=16, kernel_size=9, name='L4'),
        lc2d_layer(filters=16, kernel_size=7, strides=2, name='L5'),
        lc2d_layer(filters=16, kernel_size=5, name='L6'),
        keras.layers.Flatten(name='F0'),
        dense_layer(units=4096, activation=keras.activations.relu, name='F7'),
        keras.layers.Dropout(0.3, name='D0'),
        dense_layer(units=NUM_CLASSES, activation=keras.activations.softmax, name='F8')
    ], name='DeepFace')
    deepface.summary()

    """
    A tf.keras.optimizers.SGD will
    be used for training,
    and compile the model
    """
    sgd_opt = keras.optimizers.SGD(lr=LEARN_RATE, momentum=MOMENTUM)
    cce_loss = keras.losses.categorical_crossentropy

    deepface.compile(optimizer=sgd_opt, loss=cce_loss, metrics=['accuracy'])
    return deepface

# Prepare the data pipeline
# for train, val images
import dataset
train, val = dataset.get_train_test_dataset(CL_PATH, DATASET_PATH, IMAGE_SIZE, BATCH_SIZE)
# these are essential values that have to be set
# in order to determine the right number of steps per epoch
train_samples, val_samples = 2307424, 25893
dataset.SHUFFLE_BUFFER = train_samples
assert train.num_classes == val.num_classes == NUM_CLASSES

# Add some tf.keras.callbacks.Callback(s)
# to enhance op(s)
# like TensorBoard visualisation
# and ReduceLR
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
    patience=1, min_lr=0.0001, verbose=1)
tensorboard = keras.callbacks.TensorBoard(TB_PATH)
checkpoints = keras.callbacks.ModelCheckpoint('weights.{epoch:02d}_{val_acc:.4f}.hdf5',
    monitor='val_acc', save_weights_only=True)

cbs = [reduce_lr, checkpoints, tensorboard]

with strategy.scope():
    deepface = create_deepface()

train_history = deepface.fit(train.data, steps_per_epoch=train_samples // BATCH_SIZE + 1,
    validation_data=val.data, validation_steps=val_samples // BATCH_SIZE + 1,
    callbacks=cbs, epochs=EPOCHS)

deepface.save('model.h5')

# Let's visualise how the
# training went
from matplotlib import pyplot as plt
def save_plots():
    """
    Show and save two
    plot(s) 
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

