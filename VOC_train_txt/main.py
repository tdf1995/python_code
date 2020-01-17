#用来根据数据集生成train.txt文件
import os
from os import walk
file_path = 'D:\python code\COCO2VOC\Annotations'
txtfile = 'D:/python code/COCO2VOC/ImageSets/Main/train.txt'
if __name__ == '__main__':
    try:
        txt = open(txtfile,'w')
    except IOError:
        print('file open error')
    for (dirpath, dirnames, filenames) in walk(file_path):
        for filename in filenames:
            txt.write(filename[:-4]+'\n')