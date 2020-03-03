#生成lenet训练所需的train_list.txt
import os
import glob
dirpath = r'E:\ocr\单字符识别\test_set'
pic_files = glob.glob(dirpath+'/*/*.jpg')
f = open(dirpath+'/test_list.txt','w')
for pic_file in pic_files:
    f.write(pic_file+'\n')