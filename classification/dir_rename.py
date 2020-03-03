# 分类训练时，文件夹名需数字编码排布，故重命名
import os
import re
txt_path = r'E:\ocr\单字符识别\clean\clean\clean.txt'# 数字与文件名对应txt
dir_path = r'E:\ocr\单字符识别\clean\clean'# 数据集根目录

names = os.listdir(dir_path)
f = open(txt_path,'r')
lines = f.readlines()
for line in lines:
    a = re.split('[: :\n:]',line)
    if a[0] in names:
        os.rename(os.path.join(dir_path,a[0]),os.path.join(dir_path,a[1]))
