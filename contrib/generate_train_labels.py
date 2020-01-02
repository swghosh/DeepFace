TRAIN_PATH = '/path/to/dataset/train'

import glob, os
os.chdir(TRAIN_PATH)

class_labels = glob.glob('*')

with open('VGGFace2-class_labels_train.txt', 'w') as txt_file:
     for cl in class_labels:
          print(cl, file=txt_file)
