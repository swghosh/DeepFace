# DeepFace

Open source implementation of the renowned publication titled ["DeepFace: Closing the Gap to Human-Level Performance in Face Verification"](https://research.fb.com/publications/deepface-closing-the-gap-to-human-level-performance-in-face-verification/) by Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, Lior Wolf published at [Conference on Computer Vision and Pattern Recognition (CVPR)](http://openaccess.thecvf.com/menu.py) 2014.

![DeepFace (CVPR14) Network Architecture](https://storage.googleapis.com/swgghosh/deep-face-architecture.png)

Implementation of this paper have been done using Keras ([tf.keras](https://www.tensorflow.org/guide/keras)). This project and necessary research was supported by the [TensorFlow Research Cloud (TFRC)](https://www.tensorflow.org/tfrc). [GCP](https://cloud.google.com) resources have been used to train a million-scale machine learning model using [Cloud TPUs](https://cloud.google.com/tpu/).


# Steps to train on a GPU

## Gathering Data
Make your account on ```http://zeus.robots.ox.ac.uk/vgg_face2/login/``` once you login goto ```http://zeus.robots.ox.ac.uk/vgg_face2/``` and download all the files mentioned there 

1. Train Data_v1. 	36G. MD5: 88813c6b15de58afc8fa75ea83361d7f.
2. Test Data_v1. 	1.9G. MD5: bb7a323824d1004e14e00c23974facd3.
3. Train_Images_v1. 	The training image list, e.g., 'n000002/0001_01.jpg'.
4. Test_Images_v1. 	The test image list.


If are using a server a quick way to get this data is hijack it using cookie, for example:
```
curl --header "cookie: " http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/meta/test_list.txt --output /home/Documents/ajinkya/output/test_images_v1.txt
```
## Generating supporting files

### Create labels
After you have all this data run ```python3 generate_train_labels.py``` make sure you enter the path of 'train' folder that you downloaded in previous step. This code will generate VGGFace2-class_labels_train.txt file which you need in next steps

### Create tf records
Goto ```prepare_tfrecords.py``` file from tfrecords-faster directory and run it make sure of the paths and generate tfrecords for both train and test



