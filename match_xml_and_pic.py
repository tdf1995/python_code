import os
import glob

xml_root = r'E:\蒸浏记\分割\原数据\单类别\xml'
pic_root = r'E:\蒸浏记\分割\原数据\单类别\pic'

xml_files = os.listdir(xml_root)
pic_files = os.listdir(pic_root)
for e in xml_files:
    if e[:-4]+'.jpg' in pic_files:
        continue
    else:
        os.remove(os.path.join(xml_root,e))
for ex in pic_files:
    if ex[:-4]+'.xml' in xml_files:
        continue
    else:
        os.remove(os.path.join(pic_root,ex))