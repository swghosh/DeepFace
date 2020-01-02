# Steps to train on a GPU using VGGFace2 dataset

## Gathering Data
Create your account on ```http://zeus.robots.ox.ac.uk/vgg_face2/login/``` once you login goto ```http://zeus.robots.ox.ac.uk/vgg_face2/``` and download all the files mentioned there.

1. Train Data_v1. 	36G. MD5: 88813c6b15de58afc8fa75ea83361d7f.
2. Train_Images_v1. 	The training image list, e.g., 'n000002/0001_01.jpg'.

## Prepare Dataset

### Create Class Labels File
After you have all this data run 
```sh
python3 generate_train_labels.py
``` 
make sure you enter the path of train folder that you downloaded in previous step. 
This code will generate VGGFace2-class_labels_train.txt file which you need in next steps.

### Create TF-Records
Goto [```prepare_tfrecords.py```](https://github.com/swghosh/tfrecords-faster/blob/master/tfrecsfaster/prepare_tfrecords.py) file from [tfrecords-faster](https://github.com/swghosh/tfrecords-faster) repository and run it make sure of the paths and generate tfrecords for both Train and Test.

## Start Training
Make sure your paths are correct on [train_on_gpu.py](./train_on_gpu.py) script:

```python
IMAGE_SIZE = (152, 152)
CHANNELS = 3
NUM_CLASSES = 8631

BATCH_SIZE = 8
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
