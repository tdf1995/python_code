# 解决多文件夹内文件同名问题,将文件夹名字加到文件名前
import os
from os import walk
from tqdm import tqdm

root_path = r'D:\python code\torch_code\dataset'

def Add_to_name(root_path):
    rootdir = os.listdir(root_path)
    for e in tqdm(rootdir):
       subdir = os.path.join(root_path,e)   #   子文件及子文件夹路径
       if os.path.isfile(subdir):   #   如果是文件
           new_filename = os.path.basename(root_path) + '_'+e
           new_filepath = os.path.join(root_path,new_filename)
           os.rename(subdir, new_filepath)
       elif os.path.isdir(subdir):  #   如果是文件夹
           Add_to_name(subdir)

if __name__ == '__main__':
    Add_to_name(root_path)