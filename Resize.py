import os
from os import walk
import cv2
import re
import shutil
import numpy as np
from math import ceil

# root_path = 'D:/test/'
# xml_path = 'C:/Users/Administrator/Desktop/class.xml'


def preprocess(root_path, output_height, output_width):
   rootdir = os.listdir(root_path)
   for e in rootdir:
       subdir = os.path.join(root_path,e)   #   子文件及子文件夹路径
       if os.path.isfile(subdir):   #   如果是文件
            if os.path.splitext(subdir)[1] in {'.jpg','.bmp','.png'}:
                img = cv2.imdecode(np.fromfile(subdir,dtype=np.uint8),-1)
                img = cv2.resize(img, (output_width, output_height), interpolation=cv2.INTER_NEAREST)
                if os.path.splitext(subdir)[1] == '.jpg':
                    cv2.imencode('.jpg', img)[1].tofile(subdir)
                if os.path.splitext(subdir)[1] == '.bmp':
                    cv2.imencode('.bmp', img)[1].tofile(subdir)
                if os.path.splitext(subdir)[1] == '.png':
                    cv2.imencode('.png', img)[1].tofile(subdir)
       elif os.path.isdir(subdir):  #   如果是路径
           preprocess(subdir, output_height, output_width)

