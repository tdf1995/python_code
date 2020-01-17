# 旨在对原图进行一些裁剪处理，更贴合感兴趣区域
# 设定左上角坐标和宽高，在原图中显示框

import os
from os import walk
import cv2
import numpy as np
from math import ceil

root_path = r'G:\test\0'
xmin = 50
ymin = 200
croped_width_rate = 0.95
croped_height_rate = 0.6
crop_flag = True

def crop_ROI(root_path,xmin , ymin, croped_height_rate, croped_width_rate, crop_flag = False):
    rootdir = os.listdir(root_path)
    for e in rootdir:
        subdir = os.path.join(root_path, e)  # 子文件及子文件夹路径
        if os.path.isfile(subdir):  # 如果是文件
            if os.path.splitext(subdir)[1] in {'.jpg', '.bmp', '.png'}:
                img = cv2.imdecode(np.fromfile(subdir, dtype=np.uint8), -1)
                height, width, _ = img.shape
                xmax = ceil(xmin + width*croped_width_rate)
                ymax = ceil(ymin + height * croped_height_rate)
                if xmax> width:
                    xmax = width
                if ymax > height:
                    ymax = height

                if crop_flag:
                    img = img[ymin:ymax, xmin:xmax]
                    if os.path.splitext(subdir)[1] == '.jpg':
                        cv2.imencode('.jpg', img)[1].tofile(subdir)
                    if os.path.splitext(subdir)[1] == '.bmp':
                        cv2.imencode('.bmp', img)[1].tofile(subdir)
                    if os.path.splitext(subdir)[1] == '.png':
                        cv2.imencode('.png', img)[1].tofile(subdir)
                else:# 只显示不裁剪
                    cv2.rectangle(img, (xmin, ymin),(xmax, ymax), (0, 0, 255), 1,16)
                    cv2.putText(img, 'ROI', (xmin + 3, ymin + 10), cv2.FONT_ITALIC, 0.4, (255, 0, 255), 1)
                    cv2.imshow('test',img)
                    cv2.waitKey(100)
        elif os.path.isdir(subdir):  # 如果是路径
            crop_ROI(subdir, xmin, ymin, croped_height_rate, croped_width_rate, crop_flag)
if __name__ == '__main__':
    crop_ROI(root_path,xmin , ymin, croped_height_rate, croped_width_rate, crop_flag)