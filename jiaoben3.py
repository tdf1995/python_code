import cv2
import os
from os import walk
import numpy as np
import random
import glob
import re


path = r'E:\玉米\数据\train_set\Annotations\xmls/'
files = glob.glob(path+'*.xml')
for file in files:
    new_file = file.replace("_玉米","")
    os.rename(file,new_file)
