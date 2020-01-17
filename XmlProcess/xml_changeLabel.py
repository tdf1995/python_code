# 该脚本用于改写xml中的label名
import cv2
import os
import glob
import xml.etree.ElementTree as ET
import numpy as np

xml_root_path = r'E:\多目标跟踪\Market-1501-v15.09.15\VOCdevkit\VOC2007\新建文件夹'
Label = '1'
def changeLabel(xml_root_path):
    rootdir = os.listdir(xml_root_path)
    for e in rootdir:
        subdir = os.path.join(xml_root_path, e)  # 子文件及子文件夹路径
        if os.path.isfile(subdir):
            if os.path.splitext(subdir)[1] == '.xml':
                print(subdir)
                tree = ET.parse(subdir)
                root = tree.getroot()
                objs = root.findall('object')
                for obj in objs:
                    obj.find('name').text = Label
                tree.write(subdir)

        elif os.path.isdir(subdir):  #   如果是路径
            changeLabel(subdir)



if __name__ == '__main__':
    changeLabel(xml_root_path)