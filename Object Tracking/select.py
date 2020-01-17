# 每隔几帧挑选一张图像
import os
import shutil
import glob

pic_path = r'E:\多目标跟踪\新建文件夹'
target_path = r'E:\多目标跟踪\检测训练挑选'
pic_files = glob.glob(pic_path+'/*.jpg')
for i,file in enumerate(pic_files):
    if i%10 ==0:
        shutil.move(file, os.path.join(target_path, os.path.basename(file)))
