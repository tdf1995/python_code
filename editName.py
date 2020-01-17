import os
from os import walk
import re

root_path = r'E:\条烟\阜阳新采集图像测试_条烟分类_人工审核1122\挑选测试/'

if __name__ == '__main__':
    for (dirpath, dirnames,_) in walk(root_path):
        for dirname in dirnames:
            name = re.split('[.]',dirname)
            os.rename(os.path.join(root_path, dirname),os.path.join(root_path, name[0]))
        # for dirname in dirnames:
        #     direpath = os.path.join(root_path, dirname)
        #     path_list = os.listdir(direpath)
        #     path_list.sort()
        #     for filename in path_list:
        #         a = re.split('[_:.]',filename)
        #         # print(a[0],a[1],a[2])
        #         a[0]=a[0].zfill(4)
        #         a[1] = a[1].zfill(4)
        #         a[2]=a[2].zfill(6)
        #         # print(a[0], a[1], a[2])
        #         newname = a[0]+'_'+a[1]+'_'+a[2]
        #         print(newname)
        #         os.rename(os.path.join(direpath, filename),os.path.join(direpath, newname)+'.bmp')