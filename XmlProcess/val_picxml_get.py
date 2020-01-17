# 目的在于将val.txt中的xml(pic)文件提取出来
import os
from os import walk
import re
import shutil

val_path = 'D:/tdf/models/research/object_detection/VOCtrainval_11-May-2012/VOCdevkit/VOC2007/ImageSets/Main/val.txt'
xml_path = 'D:/tdf/models/research/object_detection/VOCtrainval_11-May-2012/VOCdevkit/VOC2007/Annotations/'
pic_path = 'D:/tdf/models/research/object_detection/VOCtrainval_11-May-2012/VOCdevkit/VOC2007/JPEGImages/'
val_xml_output = 'D:/test/xml/'
val_pic_output = 'D:/test/pic/'


if __name__ == '__main__':
    try:
        valtext = open(val_path,'r')
    except IOError:
        print('file open error:')
    txt = valtext.read()
    FileNames = re.split('[\n: ]',txt)
    for FileName in FileNames:
        if FileName != '\n':
            xmlName = FileName + '.xml'
            picName = FileName + '.jpg'
            if os.path.exists(os.path.join(xml_path,xmlName)) and os.path.exists(os.path.join(pic_path,picName)):
                shutil.copyfile(os.path.join(xml_path,xmlName),os.path.join(val_xml_output,xmlName))
                shutil.copyfile(os.path.join(pic_path, picName), os.path.join(val_pic_output, picName))