import os
import glob

voc_path = r'D:\python code\torch_code\dataset' #voc格式数据根路径
f = open(r'D:\python code\torch_code\dataset.txt','w')
for (dirpath, dirnames, _)in os.walk(voc_path):
    for dirname in dirnames:
        dir_path = os.path.join(voc_path,dirname)
        for (_, _, filenames) in os.walk(dir_path):
            for filename in filenames:
                f.write(filename+' '+dirname+'\n')
