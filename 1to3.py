import numpy as np
import os
import glob
import cv2

path = r'E:\ocr\dataset\新增'
files = glob.glob(path+'\*.jpg')
for file in files:
    img = cv2.imdecode(np.fromfile(file,dtype=np.uint8),-1)
    img = np.expand_dims(img, axis=2)
    img = np.concatenate((img, img, img), axis=-1)
    cv2.imencode('.jpg', img)[1].tofile(file)