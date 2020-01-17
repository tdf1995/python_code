# 该脚本用于随机裁剪图像
import cv2
import os
from os import walk
import numpy as np
import random

per_pic_time = 3 # 每张图像生成多少张
root_path = r'E:\细胞\x100\Pic1\Pic'
crop_ratio = 4.5 # 原图和目标裁剪图的长度之比

def random_crop(image):

    height, width,_ = image.shape
    target_height = int(height / crop_ratio)
    target_width = int(width / crop_ratio)
    ny = random.randint(0, height - target_height)
    nx = random.randint(0, width - target_width)
    image = image[ny:ny+target_height, nx:nx + target_width]

    return image

def preprocess(root_path):
   rootdir = os.listdir(root_path)
   for e in rootdir:
       subdir = os.path.join(root_path,e)   #   子文件及子文件夹路径
       if os.path.isfile(subdir):   #   如果是文件
            if os.path.splitext(subdir)[1] in {'.jpg','.bmp','.png'}:
                img = cv2.imdecode(np.fromfile(subdir,dtype=np.uint8),cv2.IMREAD_UNCHANGED)
                for i in range(per_pic_time):
                    img_crop = random_crop(img)
                    if os.path.splitext(subdir)[1] == '.jpg':
                        cv2.imencode('.jpg', img_crop)[1].tofile(subdir[:-4]+'_'+str(i)+'.jpg')
                    if os.path.splitext(subdir)[1] == '.bmp':
                        cv2.imencode('.bmp', img_crop)[1].tofile(subdir[:-4]+'_'+str(i)+'.jpg')
                    if os.path.splitext(subdir)[1] == '.png':
                        cv2.imencode('.png', img_crop)[1].tofile(subdir[:-4]+'_'+str(i)+'.jpg')
       elif os.path.isdir(subdir):  #   如果是路径
           preprocess(subdir)

if __name__ == '__main__':
    preprocess(root_path)