import cv2
import os
from os import walk
import numpy as np
from math import ceil
from tqdm import tqdm

cell_path =r'C:\Users\tdf\Desktop\细胞分割新标注'
output_height=280
output_width=280
def cell_preprocess(root_path, output_height, output_width):
    output_ratio = output_width / output_height
    rootdir = os.listdir(root_path)
    for e in tqdm(rootdir):
       subdir = os.path.join(root_path,e)   #   子文件及子文件夹路径
       if os.path.isfile(subdir):   #   如果是文件
            if os.path.splitext(subdir)[1] in {'.jpg','.bmp','.png'}:
                img = cv2.imdecode(np.fromfile(subdir, dtype=np.uint8), -1)
                height = img.shape[0]
                width = img.shape[1]
                if height>320 or width>320:
                    ratio = width / height
                    if ratio >= output_ratio:
                        width = output_width
                        height = int(width / ratio)
                    else:
                        height = output_height
                        width = int(height * ratio)

                    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
                    img = cv2.copyMakeBorder(img, int((output_height - height) / 2), ceil((output_height - height) / 2),
                                             int((output_width - width) / 2), ceil((output_width - width) / 2), cv2.BORDER_CONSTANT,
                                             value=[0, 0, 0])
                else:
                    img = cv2.copyMakeBorder(img, int((output_height - height) / 2), ceil((output_height - height) / 2),
                                             int((output_width - width) / 2), ceil((output_width - width) / 2), cv2.BORDER_CONSTANT,
                                             value=[0, 0, 0])
                if os.path.splitext(subdir)[1] == '.jpg':
                    cv2.imencode('.jpg', img)[1].tofile(subdir)
                if os.path.splitext(subdir)[1] == '.bmp':
                    cv2.imencode('.bmp', img)[1].tofile(subdir)
                if os.path.splitext(subdir)[1] == '.png':
                    cv2.imencode('.png', img)[1].tofile(subdir)
       elif os.path.isdir(subdir):  #   如果是文件夹
           cell_preprocess(subdir, output_height, output_width)

if __name__ == '__main__':
    cell_preprocess(cell_path, output_height, output_width)