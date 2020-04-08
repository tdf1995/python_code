import os
from os import walk
import glob

pic_path = r'E:\project\pedestrian\source dateset\VOC2007\JPEGImages'
txt_path = r'E:\project\pedestrian\source dateset\VOC2007\ImageSets\Main\train.txt'

files = glob.glob(r'E:\project\pedestrian\source dateset\VOC2007\JPEGImages\*.jpg')
f = open(txt_path,'w')
for file in files:
    # f.write(os.path.join('E:\ocr\单字符识别\dataset/',os.path.basename(os.path.dirname(file)),os.path.basename(file))+'\n')
    f.write(os.path.basename(file)[:-4] + '\n')
f.close()