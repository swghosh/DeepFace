# DeepFace

Open source implementation of the renowned publication titled ["DeepFace: Closing the Gap to Human-Level Performance in Face Verification"](https://research.fb.com/publications/deepface-closing-the-gap-to-human-level-performance-in-face-verification/) by Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, Lior Wolf published at Conference on [Computer Vision and Pattern Recognition (CVPR)](http://openaccess.thecvf.com/menu.py) 2014.

The proposed CNN architecture in the paper have been implemented using Keras ([tf.keras](https://www.tensorflow.org/guide/keras)) and have been trained on Cloud TPUs. 

![DeepFace (CVPR14) Network Architecture](https://storage.googleapis.com/swgghosh/deep-face-architecture.png)

This project and necessary research was supported by the [TensorFlow Research Cloud (TFRC)](https://www.tensorflow.org/tfrc). We're grateful to Google for providing us all the [GCP](https://cloud.google.com) resources to make million scale machine learning models training possible using the state of the art AI hardware like [Cloud TPUs](https://cloud.google.com/tpu/).

Pre-trained weights on the VGGFace2 dataset are available here.
