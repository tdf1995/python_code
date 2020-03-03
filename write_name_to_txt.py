import os
from os import walk
import glob

pic_path = r'E:\ocr\检测\原图mask分割后的单个字符检测\VOC2007\JPEGImages'
txt_path = r'E:\ocr\检测\原图mask分割后的单个字符检测\VOC2007\Main\train.txt'

files = glob.glob(r'E:\ocr\检测\原图mask分割后的单个字符检测\VOC2007\JPEGImages\*.jpg')
f = open(txt_path,'w')
for file in files:
    f.write(os.path.basename(file)[:-4]+'\n')
    # f.write(file + '\n')
f.close()