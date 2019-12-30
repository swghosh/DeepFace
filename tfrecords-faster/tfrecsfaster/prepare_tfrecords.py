#!/usr/bin/env python3

import os
WORKERS = os.cpu_count()
print("Using %d workers." % WORKERS)
# folder structure
# tfrecords
#  - train
#  - test
dataset_path = 'path_to/testimages' #testimages contains all the testing images
tfrecs_path = '/path_to/tfrecords/test' #'tfrecords' is a folder name. 'test' is another folder inside tfrecord.  
shard_size = 12000

n_samples = None

def tensorflow():
    import tensorflow as tf
    tf.enable_eager_execution()
    return tf

def init_task():
    print('[init_task] Started...')
    tf = tensorflow()
    image_paths = tf.io.gfile.glob(dataset_path + '/*/*.jpg')
    batch_size = len(image_paths) // WORKERS
    if len(image_paths) % WORKERS != 0:
        batch_size += 1
    image_paths_batched = [
        image_paths[index * batch_size: (index + 1) * batch_size]
        for index in range(WORKERS)
    ]
    global n_samples
    n_samples = len(image_paths)
    print('[init_task] Exiting...')
    return image_paths_batched

def task(inputs):
    worker_id, filenames = inputs
    
    def print_info(text):
        print('[Worker %d] %s' % (worker_id, text))
    
    print_info('Started...')

    tf = tensorflow()

    def wrap_bytes(list_of_bytestrings):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))
    def get_image_and_class(image_path):
        class_label = tf.strings.split(tf.expand_dims(image_path, axis=-1), sep='/').values[-2]
        img = tf.io.read_file(image_path)
        return img, class_label
    def to_tfrecord(image, class_name):
        feature = {
            'image': wrap_bytes([image]),
            'class': wrap_bytes([class_name])
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))
    
    filenames = tf.convert_to_tensor(filenames)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.shuffle(n_samples // WORKERS)
    dataset = dataset.map(get_image_and_class, tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(shard_size)

    for shard_id, (image, label) in enumerate(dataset):
        print_info('Working on shard %d' % shard_id)
        image, label = image.numpy(), label.numpy()
        current_shard_limit = image.shape[0]
        filename = tfrecs_path + '/worker{}-{:06d}-l{}.tfrec'.format(worker_id, shard_id, current_shard_limit)
        with tf.io.TFRecordWriter(filename) as out_file:
            for i in range(current_shard_limit):
                record = to_tfrecord(image[i], label[i])
                out_file.write(record.SerializeToString())
        print_info('Wrote file {} containing {} records'.format(filename, current_shard_limit))

image_paths_batched = init_task()
inputs = [(index, batch) for index, batch in enumerate(image_paths_batched)]

import multiprocessing

if __name__ == '__main__':
    with multiprocessing.Pool(WORKERS) as pool:
        pool.map(task, inputs)

    print('All workers exited')
