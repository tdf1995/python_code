import cv2
import os
from os import walk
import imgaug as ia
from imgaug import  augmenters as iaa
picPath = 'D:\orgin\Faster-RCNN-TensorFlow-Python3.5-master\Faster-RCNN-TensorFlow-Python3.5-master\data\VOCDevkit2007\VOC2007\JPEGImages'
TxtFile = 'D:\Data.txt'

for (dirpath, dirnames, filenames) in walk(picPath):
    for filename in filenames:  # 文件名
        im_file = os.path.join(picPath, filename)  # 绝对地址
        img = cv2.imread(im_file)
        size = (448, 448)
        img = cv2.resize(img, size)
        cv2.imwrite(picPath+'/'+filename,img)