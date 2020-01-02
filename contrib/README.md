# Steps to train on a GPU using VGGFace2 dataset

## Gathering Dataset
Create your account on ```http://zeus.robots.ox.ac.uk/vgg_face2/login/``` once you login goto ```http://zeus.robots.ox.ac.uk/vgg_face2/``` and download all the files mentioned there.

1. Train Data_v1. 	36G. MD5: 88813c6b15de58afc8fa75ea83361d7f.
2. Train_Images_v1. 	The training image list, e.g., 'n000002/0001_01.jpg'.

## Prepare Dataset

### Create Class Labels File
For the train and test data pipeline, we have to generate a text file containing line-seperated class labels.

Please make sure to set the correct path of the train folder in [generate_train_labels.py](./generate_train_labels.py) script

```sh
python3 generate_train_labels.py
``` 

This script will generate `VGGFace2-class_labels_train.txt` file which is needed further.

### Create TF-Records
Use the [```prepare_tfrecords.py```](https://github.com/swghosh/tfrecords-faster/blob/master/tfrecsfaster/prepare_tfrecords.py) script from [tfrecords-faster](https://github.com/swghosh/tfrecords-faster) repository to generate TFRecords. 

Please make sure that the paths are correct for train dataset.

## Start Training
Also make sure that the paths are correct in [train_on_gpu.py](./train_on_gpu.py) script:

```python
IMAGE_SIZE = (152, 152)
CHANNELS = 3
NUM_CLASSES = 8631 # VGGFace_train_v1 dataset

BATCH_SIZE = 8 # tune it as per your GPU memory
LEARN_RATE = 0.01 * (BATCH_SIZE / 128)
MOMENTUM = 0.9
EPOCHS = 15

CL_PATH = './VGGFace2-class_labels_train.txt'
DATASET_PATH = './tfrecords'
TB_PATH = './tensorboard_logs'
```

Then, run:
```sh
python3 train_on_gpu.py
```
