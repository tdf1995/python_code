import os
import random
import glob

pic_Path = 'F:\models-master\cell\VOCdevkit\VOC2007\JPEGImages'
xml_Path = 'F:\models-master\cell\VOCdevkit\VOC2007\Annotations'

picList = os.listdir(pic_Path)
random.shuffle(picList)
total_num = len(picList)
xmlList = glob.glob('F:\models-master\cell\VOCdevkit\VOC2007\Annotations\*.xml')

for i in range(total_num):
    pic = picList[i]
    pic_path = os.path.join(pic_Path, pic)
    xml_path = os.path.join(xml_Path, pic[:-4]+'.xml')
    if xml_path in xmlList:
        pic = ("%05d" % i) + '.jpg'
        new_pic_path = os.path.join(pic_Path, pic)
        new_xml_path = os.path.join(xml_Path, pic[:-4]+'.xml')
        os.rename(pic_path, new_pic_path)
        os.rename(xml_path, new_xml_path)
