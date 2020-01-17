import os
from os import walk
import glob

pic_path = r'E:\多目标跟踪\cosine_metric_learning-master\jpg'
txt_path = r'E:\多目标跟踪\cosine_metric_learning-master\info\train_name.txt'

files = glob.glob(r'E:\多目标跟踪\cosine_metric_learning-master\jpg\*.jpg')
f = open(txt_path,'w')
for file in files:
    f.write(os.path.basename(file)+'\n')
    # f.write(file + '\n')
f.close()