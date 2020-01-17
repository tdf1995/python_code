'''
这个脚本用来将图像按原长宽比缩放到规定尺寸
然后padding到统一尺寸
'''
import cv2
import os
from os import walk
import numpy as np
from math import ceil
from tqdm import tqdm

def preprocess(root_path, output_height, output_width):
    output_ratio = output_width / output_height
    rootdir = os.listdir(root_path)
    for e in tqdm(rootdir):
       subdir = os.path.join(root_path,e)   #   子文件及子文件夹路径
       if os.path.isfile(subdir):   #   如果是文件
            if os.path.splitext(subdir)[1] in {'.jpg','.bmp','.png'}:
                img = cv2.imdecode(np.fromfile(subdir, dtype=np.uint8), -1)
                (height, width, _) = img.shape
                ratio = width / height
                if ratio >= output_ratio:
                    width = output_width
                    height = int(width / ratio)
                else:
                    height = output_height
                    width = int(height * ratio)
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
                img = cv2.copyMakeBorder(img, int((output_height - height) / 2), ceil((output_height - height) / 2),
                                         int((output_width - width) / 2), ceil((output_width - width) / 2), cv2.BORDER_CONSTANT,
                                         value=[0, 0, 0])
                if os.path.splitext(subdir)[1] == '.jpg':
                    cv2.imencode('.jpg', img)[1].tofile(subdir)
                if os.path.splitext(subdir)[1] == '.bmp':
                    cv2.imencode('.jpg', img)[1].tofile(subdir)
                if os.path.splitext(subdir)[1] == '.png':
                    cv2.imencode('.jpg', img)[1].tofile(subdir)
       elif os.path.isdir(subdir):  #   如果是文件夹
           preprocess(subdir, output_height, output_width)


# if __name__ == '__main__':
#     G_width = 625
#     G_height = 200
#     G_ratio = G_width / G_height
#
#     for (_,dirnames,_)in walk(root_path):
#         for dirname in dirnames:
#             dirpath = os.path.join(root_path, dirname)
#             for (_,direnames,_)in walk(dirpath):
#                 for direname in direnames:
#                     direpath = os.path.join(dirpath, direname)
#                     for (_,_,filenames)in walk(direpath):
#                         for filename in filenames:
#                             img = cv2.imdecode(np.fromfile(os.path.join(direpath,filename),dtype=np.uint8),-1)
#                             (height,width,_) = img.shape
#                             ratio = width/height
#                             if ratio>=G_ratio:
#                                 width = G_width
#                                 height = int(width/ratio)
#                             else:
#                                 height = G_height
#                                 width = int(height*ratio)
#                             img = cv2.resize(img,(width,height),interpolation=cv2.INTER_NEAREST)
#                             img = cv2.copyMakeBorder(img, int((G_height - height) / 2), ceil((G_height - height) / 2),int((G_width - width) / 2), ceil((G_width - width) / 2),cv2.BORDER_CONSTANT, value=[0, 0, 0])
#                             outpath = os.path.join(direpath,filename)
#                             cv2.imencode('.jpg',img)[1].tofile(outpath)