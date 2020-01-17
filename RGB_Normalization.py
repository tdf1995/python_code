# 对图像进行RGB三通道归一化操作
import os
from os import walk
import cv2
import numpy as np
from math import ceil

root_path = r'E:\喷码\裁剪+归一化'

def RGB_Normalized(root_path):
    rootdir = os.listdir(root_path)
    for e in rootdir:
        subdir = os.path.join(root_path, e)  # 子文件及子文件夹路径
        if os.path.isfile(subdir):  # 如果是文件
            if os.path.splitext(subdir)[1] in {'.jpg', '.bmp', '.png'}:
                img = cv2.imdecode(np.fromfile(subdir, dtype=np.float32), -1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # cv2.imshow('2', img[:, :, 0])
                img[:,:,0] = img[:,:,0] - 123.68
                img[:, :, 0] = img[:, :, 0] / 58.393
                img[:, :, 1] = img[:, :, 1] - 116.779
                img[:, :, 1] = img[:, :, 1] / 57.12
                img[:, :, 2] = img[:, :, 2] - 103.939
                img[:, :, 2] = img[:, :, 2] / 57.375
                if os.path.splitext(subdir)[1] == '.jpg':
                    cv2.imencode('.jpg', img)[1].tofile(subdir)
                if os.path.splitext(subdir)[1] == '.bmp':
                    cv2.imencode('.bmp', img)[1].tofile(subdir)
                if os.path.splitext(subdir)[1] == '.png':
                    cv2.imencode('.png', img)[1].tofile(subdir)
            else:
                continue
        elif os.path.isdir(subdir):  # 如果是路径
            RGB_Normalized(subdir)
if __name__ == '__main__':
    RGB_Normalized(root_path)