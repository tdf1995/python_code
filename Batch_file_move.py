#大批量同后缀文件移动
import os
from os import walk
import time
import shutil
from numba import jit


root_path = r'C:\Users\tdf\Desktop\条烟分割库_2019_0304_3_删除小目标标注\条烟分割库_2019_0304_3_删除小目标标注'
target_path = r'E:\条烟\分割\数据\xml'
suffix = '.xml'
mode = 0#0为复制1为剪切

@jit
def Batch_move(root_path, target_path, suffix, mode = 0):
   rootdir = os.listdir(root_path)
   for e in rootdir:
       subdir = os.path.join(root_path,e)   #   子文件及子文件夹路径
       if os.path.isfile(subdir):   #   如果是文件
            if os.path.splitext(subdir)[1] == suffix:
                print(subdir)
                if mode == 0:
                    shutil.copy(subdir, os.path.join(target_path, os.path.basename(subdir)))
                elif mode == 1:
                    shutil.move(subdir, os.path.join(target_path, os.path.basename(subdir)))
                else:
                    print('Mode must be 0 or 1!')
                    os._exit(0)
       elif os.path.isdir(subdir):  #   如果是路径
           Batch_move(subdir, target_path, suffix, mode)

if __name__ == '__main__':
    time_start = time.time()
    Batch_move(root_path, target_path, suffix, mode)
    time_end = time.time()
    print('总耗时%f'%(time_end-time_start))
