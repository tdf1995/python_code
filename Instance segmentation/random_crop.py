# 该脚本用于随机裁剪图像
import cv2
import os
from os import walk
import numpy as np
import random


root_path = r'E:\黄豆\增强\背景\原图'
target_path = r'E:\黄豆\增强\背景\裁剪'

def preprocess(root_path):
   rootdir = os.listdir(root_path)
   for e in rootdir:
       subdir = os.path.join(root_path,e)   #   子文件及子文件夹路径
       if os.path.isfile(subdir):   #   如果是文件
            if os.path.splitext(subdir)[1] in {'.jpg','.bmp','.png'}:
                img = cv2.imdecode(np.fromfile(subdir,dtype=np.uint8),cv2.IMREAD_UNCHANGED)
                height, width, _ = img.shape
                random_ys = random.randint(0,2448-416)
                random_xs = random.randint(0,3264-416)
                xe = random_xs + 416
                ye = random_ys + 416
                im = img[random_ys:ye,random_xs:xe]
                cv2.imencode('.jpg', im)[1].tofile(os.path.join(target_path,os.path.basename(subdir))[:-4] + '_crop_'+str(random_xs)+'_'+str(random_ys)+ '.jpg')

       elif os.path.isdir(subdir):  #   如果是路径
           preprocess(subdir)

if __name__ == '__main__':
    preprocess(root_path)