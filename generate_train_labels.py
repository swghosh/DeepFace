import glob
class_labels = glob.glob('path_to/train/*')

with open('VGGFace2-class_labels_train.txt', 'w') as txt_file:
     for cl in class_labels:
          print(cl[58:], file=txt_file)   # 58 must me changed according to your string characters in the current working directory

#for file in class_labels:
#	print(file[58:])
