import cv2
import glob
import os
import numpy as np

pic_root_path = r'E:\多目标跟踪\pic\2'
pic_files = glob.glob(pic_root_path+'\*.jpg')
target_path = r'E:\多目标跟踪\pic_crop/'
for pic_file in pic_files:
    pic = cv2.imdecode(np.fromfile(pic_file, dtype=np.uint8), -1)
    pic_crop = pic[280:640, 0:360]
    # cv2.imshow('1',pic_crop)
    # cv2.waitKey(0)
    cv2.imencode('.jpg', pic_crop)[1].tofile(os.path.join(target_path, os.path.basename(pic_file)))