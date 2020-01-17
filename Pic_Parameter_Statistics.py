# 统计一批图像中的长、宽、比例等参数
import os
import cv2
import numpy as np


root_path = r'E:\菜品\分割\1111'

max_height = 0
max_width = 0
max_ratio = 0
min_height = 999999999
min_width = 999999999
min_ratio = 999999999


def preprocess(root_path):
   global max_height
   global max_width
   global max_ratio
   global min_height
   global min_width
   global min_ratio
   rootdir = os.listdir(root_path)
   for e in rootdir:
       subdir = os.path.join(root_path,e)   #   子文件及子文件夹路径
       if os.path.isfile(subdir):   #   如果是文件
            if os.path.splitext(subdir)[1] in {'.jpg','.bmp','.png'}:
                img = cv2.imdecode(np.fromfile(subdir,dtype=np.uint8),cv2.IMREAD_UNCHANGED)
                if img is not None:
                    height = img.shape[0]
                    width = img.shape[1]
                    ratio = height/width
                    if height > max_height:
                        max_height = height
                        print('max_height:',subdir)
                    if height < min_height:
                        min_height = height
                        print('min_height:',subdir)
                    if width > max_width:
                        max_width = width
                        print('max_width:',subdir)
                    if width < min_width:
                        min_width = width
                        print('min_width:',subdir)
                    if ratio > max_ratio:
                        max_ratio = ratio
                        print('max_ratio:',subdir)
                    if ratio < min_ratio:
                        min_ratio = ratio
                        print('min_ratio:',subdir)
       elif os.path.isdir(subdir):  #   如果是路径
           preprocess(subdir)

if __name__ == '__main__':
    preprocess(root_path)
    print('max_height:', max_height)
    print('max_width:', max_width)
    print('min_height:', min_height)
    print('min_width:', min_width)
    print('max_ratio:', max_ratio)
    print('min_ratio:', min_ratio)