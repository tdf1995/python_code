'''
这个脚本在于统计条烟数据库内烟的种类和对应的数量
'''

import xlwt
import re
import os
from os import walk
import DataProcess.XML as XML
from xml.dom import minidom

root_path = 'D:/条烟分类库_1009/'
xml_path = 'C:/Users/Administrator/Desktop/class.xml'

if __name__ == '__main__':
    dataset = xlwt.Workbook(encoding = 'utf-8', style_compression=0)
    for(dirpath,dirnames,_)in walk(root_path):
        for dirname in dirnames:
            sheet = dataset.add_sheet(dirname)
            sheet.write(0, 0, '烟标签')
            sheet.write(1, 0, '烟名')
            sheet.write(2, 0, '数量')
            direpath = os.path.join(root_path,dirname)
            for(_,labels,_)in walk(direpath):
                i = 0
                for label in labels:
                    i = i+1
                    name = []
                    sheet.write(0, i, label)
                    dom = minidom.parse(xml_path)
                    dom, objs = XML.splitXml(dom)
                    for j in range(objs.length):
                        Label = objs[j].getElementsByTagName("label")[0].childNodes[0].nodeValue
                        Name = objs[j].getElementsByTagName("name_list")[0].childNodes[0].nodeValue
                        # name = name.replace("）",")")
                        # name = name.replace("（", "(");
                        if label== Label:
                            name = Name
                    sheet.write(1, i, name)
                    pic_path = os.path.join(direpath,label)
                    for (_,_,filenames)in walk(pic_path):
                        num = len(filenames)
                        sheet.write(2, i, num)
    dataset.save('tiaoyan_dataset.xls')