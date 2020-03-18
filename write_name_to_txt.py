import os
from os import walk
import glob

pic_path = r'E:\ocr\单字符识别\test_set'
txt_path = r'E:\ocr\单字符识别\test_set\test_list.txt'

files = glob.glob(r'E:\ocr\单字符识别\test_set\*\*.jpg')
f = open(txt_path,'w')
for file in files:
    f.write(os.path.join('E:\ocr\单字符识别\dataset/',os.path.basename(os.path.dirname(file)),os.path.basename(file))+'\n')
    # f.write(os.path.basename(file) + '\n')
f.close()