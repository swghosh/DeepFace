import tensorflow as tf

SHUFFLE_BUFFER = 1 # set this to the number of items in dataset for ideal results

class Dataset:
    @staticmethod
    def preprocess_image(img):
        img /= 255.0
        return img

    @staticmethod
    def get_class_labels(file_path):
        line_seperator = '\n' # check this carefully, causes errors in many cases!, '\r\n' vs '\n'

        file_contents = tf.io.read_file(file_path)
        file_contents = tf.expand_dims(file_contents, axis=-1)

        class_labels = tf.strings.split(file_contents, sep=line_seperator)
        class_labels = class_labels.values[:-1] # ignore empty last line
        return class_labels

    def __init__(self, cl_path, dataset_path, image_size, batch_size, shuffle=True):
        self.dataset_path, self.image_size, self.batch_size, self.shuffle = [value for value in (dataset_path, image_size, batch_size, shuffle)] # use shuffle only with train, not with test
        
        self.class_labels = Dataset.get_class_labels(cl_path)
        # the number of classes present should 
        # be evaluated in advance :(
        with tf.Session() as sess:
            self.num_classes = sess.run(tf.shape(self.class_labels)[0])
        self.data = self.get_dataset()

    def get_image_and_class(self, image, classl):
        classl = tf.math.equal(self.class_labels, classl)
        classl = tf.cast(classl, tf.int32)
        classl = tf.argmax(classl, axis=-1)
        classl = tf.one_hot(classl, self.num_classes)

        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize_image_with_pad(image, self.image_size[0], self.image_size[1])
        image = tf.cast(image, tf.float32)
        image = Dataset.preprocess_image(image)

        return image, classl

    def read_tfrec(self, example):
        feature = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'class': tf.io.FixedLenFeature([], tf.string)
        }
        example = tf.parse_single_example(example, feature)
        return self.get_image_and_class(example['image'], example['class'])
    
    def get_dataset(self):
        cycle_length = 32
        prefetch_size = 1

        option = tf.data.Options()
        option.experimental_deterministic = False

        ds = tf.data.Dataset.list_files(self.dataset_path + '/*.tfrec')
        ds = ds.with_options(option)
        ds = ds.interleave(tf.data.TFRecordDataset, cycle_length=cycle_length, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.map(self.read_tfrec, tf.data.experimental.AUTOTUNE)
        if self.shuffle:
            ds = ds.shuffle(SHUFFLE_BUFFER)
        ds = ds.repeat()
        ds = ds.batch(self.batch_size, drop_remainder=True)
        return ds.prefetch(prefetch_size)

def get_train_test_dataset(cl_path, dataset_path, image_size, batch_size):
    train_path = '/train'
    test_path = '/test'
    train, test = [
        Dataset(cl_path, dataset_path + curr_path, image_size, batch_size, training)
        for curr_path, training in zip((train_path, test_path), (True, False))
    ]
    return train, test

