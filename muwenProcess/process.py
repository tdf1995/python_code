import cv2
import matplotlib.pyplot as plt
import os
from os import walk
import numpy as np
import shutil\


pic_path = r'E:\木材\数据库\木材\木材均衡图/'
def Pic_To_Gray(root_path, out_path):
    rootdir = os.listdir(root_path)
    for e in rootdir:
        subdir = os.path.join(root_path, e)
        if os.path.isfile(subdir):
            if os.path.splitext(subdir)[1] in {'.jpg','.bmp','.png'}:
                img = cv2.imdecode(np.fromfile(subdir,dtype=np.uint8),-1)
                try:
                    img.shape
                except:
                    os.remove(subdir)
                    continue
                if (len(img.shape) == 3):
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if os.path.splitext(subdir)[1] == '.jpg':
                    cv2.imencode('.jpg', img)[1].tofile(os.path.join(out_path, e))
                if os.path.splitext(subdir)[1] == '.bmp':
                    cv2.imencode('.jpg', img)[1].tofile(os.path.join(out_path, e))
                if os.path.splitext(subdir)[1] == '.png':
                    cv2.imencode('.jpg', img)[1].tofile(os.path.join(out_path, e))
        elif os.path.isdir(subdir):  # 如果是路径
            Pic_To_Gray(subdir, subdir)

def Pic_EqualizeHist(root_path, out_path):
    rootdir = os.listdir(root_path)
    for e in rootdir:
        subdir = os.path.join(root_path, e)
        if os.path.isfile(subdir):
            if os.path.splitext(subdir)[1] in {'.jpg', '.bmp', '.png'}:
                img = cv2.imdecode(np.fromfile(subdir, dtype=np.uint8), -1)
                img = cv2.equalizeHist(img)
                if os.path.splitext(subdir)[1] == '.jpg':
                    cv2.imencode('.jpg', img)[1].tofile(os.path.join(out_path, e))
                if os.path.splitext(subdir)[1] == '.bmp':
                    cv2.imencode('.jpg', img)[1].tofile(os.path.join(out_path, e))
                if os.path.splitext(subdir)[1] == '.png':
                    cv2.imencode('.jpg', img)[1].tofile(os.path.join(out_path, e))
        elif os.path.isdir(subdir):  # 如果是路径
            Pic_EqualizeHist(subdir, subdir)
# for (_,__,filenames) in walk(pic_path):
#     for filename in filenames:
#         image = cv2.imread(pic_path+filename, cv2.IMREAD_UNCHANGED)
#         pic1 = cv2.equalizeHist(image[:,:,0])
#
#         pic2 = cv2.equalizeHist(image[:,:,1])
#
#         pic3 = cv2.equalizeHist(image[:,:,2])
#
#         im=cv2.merge([pic1,pic2,pic3])
#         cv2.imwrite(out_path+filename, im)


if __name__ == '__main__':
    # Pic_To_Gray(pic_path, pic_path)
    Pic_EqualizeHist(pic_path,pic_path)