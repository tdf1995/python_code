# 用于分类网络，根据标签文件修改类别文件夹名
import os
from os import walk
import xml.etree.ElementTree as ET
import DataProcess.XML as XML
from xml.dom import minidom
import re

xml_path = None
pic_path = 'F:\中药材/'
Label_id_path = r'C:\Users\tdf\Desktop/label.txt'

def XML_id_change():
    file_name = os.listdir(pic_path)
    dom = minidom.parse(xml_path)
    dom, objs = XML.splitXml(dom)
    for i in range(objs.length):
        label = objs[i].getElementsByTagName("label")[0].childNodes[0].nodeValue

        name = objs[i].getElementsByTagName("name_list")[0].childNodes[0].nodeValue
        # name = name.replace("）",")")
        # name = name.replace("（", "(");
        if name in file_name:
            os.rename(pic_path+name, pic_path + str(label))

def TXT_id_change():
    f = open(Label_id_path,'r')
    file_name = os.listdir(pic_path)
    lines = f.readlines()
    for line in lines:
        a = re.split('[=:,:\n:]',line)
        if a[0][1:][:-1] in file_name:
            os.rename(pic_path+a[0][1:][:-1],pic_path+a[1])

if __name__ == '__main__':
    if xml_path:
        XML_id_change()
    elif Label_id_path:
        TXT_id_change()






        # for (dirpath, dirnames, filenames) in walk(pic_path):
        #     print('1')