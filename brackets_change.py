import os
Path = r'E:\乌冬面\数据\train_set\images'
files = os.listdir(Path)
for e in files:
    if ' 'in e:
        e_new = ''.join(e.split())
        os.rename(os.path.join(Path,e), os.path.join(Path,e_new))