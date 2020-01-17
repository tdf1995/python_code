# 这个程序是用来修改xml文件的图片路径的
from xml.dom import minidom
import xml.etree.ElementTree as ET
from os import walk
import os
import codecs

xml_Path = r'E:\多目标跟踪\Market-1501-v15.09.15\VOCdevkit\VOC2007\已标注\Annotations/'
new_path = r'E:\乌冬面\数据\train_set\images/'
new_folder = 'VOC2007'
# def ChangeXmlPath(dom):
#     path = dom.getElementsByTagName("path")
#     print(path)
#     print(path[0].childNodes[0].nodeValue)
#     path[0].childNodes[0].nodeValue = 'D:/'
# # for root, dirs, Files in os.walk(Path):
# #     for dir in dirs:
# #         Sub_Path = os.path.join(Path, dir)
# #         print(Sub_Path)
for subroot, subdirs, SubFiles in os.walk(xml_Path):
    for xmlfile in SubFiles:
        if not xmlfile[-4:] == '.xml':
            continue
        tree = ET.parse(os.path.join(xml_Path, xmlfile))
        root = tree.getroot()
        # path = root.find('path')
        # path.text = new_path + xmlfile[:-4] + ".jpg"
        # folder = root.find('folder')
        # folder.text = new_folder

        objs = root.findall('object')
        for obj in objs:
            label = obj.find('name')
            label.text = '1'

        tree.write(os.path.join(xml_Path, xmlfile))