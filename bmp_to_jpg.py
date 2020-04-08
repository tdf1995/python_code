import os
from os import walk
import cv2
import re
import shutil
import numpy as np
from math import ceil




def preprocess(root_path, output_height, output_width):
   rootdir = os.listdir(root_path)
   for e in rootdir:
       subdir = os.path.join(root_path,e)   #   子文件及子文件夹路径
       if os.path.isfile(subdir):   #   如果是文件
            if os.path.splitext(subdir)[1] in {'.png'}:
                img = cv2.imdecode(np.fromfile(subdir,dtype=np.uint8),-1)
                if img is None:
                    os.remove(subdir)
                new_name = subdir[:-4] + '.jpg'
                cv2.imencode('.jpg', img)[1].tofile(new_name)
                print(subdir)
                os.remove(subdir)
       elif os.path.isdir(subdir):  #   如果是路径
           preprocess(subdir, output_height, output_width)

