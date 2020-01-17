# encoding: UTF-8
import glob as gb
import cv2
import numpy as np
img_path = gb.glob("E:\新建文件夹\*.jpg")
videoWriter = cv2.VideoWriter('test.mp4', -1, 20, (384,384))

for path in img_path:
    img  = cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
    # img = cv2.resize(img,(640,480))
    videoWriter.write(img)