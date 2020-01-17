# 该脚本用于将图像连同带有目标框信息xml文件一起缩放
import cv2
import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
import math
import os

xml_path = r'F:\models-master\cell\VOCdevkit\VOC2007\Annotations'

def xml_pic_Resize(xml_path):
    xml_files = glob.glob(xml_path+'/*.xml')
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        xml_name = os.path.basename(xml_file)
        root = tree.getroot()
        root.find('filename').text = xml_name[:-4]+'.jpg'


        # for obj in root.findall('object'):
        #     xmin = float(obj.find('bndbox').find('xmin').text)
        #     xmax = float(obj.find('bndbox').find('xmax').text)
        #     ymin = float(obj.find('bndbox').find('ymin').text)
        #     ymax = float(obj.find('bndbox').find('ymax').text)
        #
        #     # obj_center_x = float(xmin + xmax)/2
        #     # obj_center_y = float(ymin + ymax)/2
        #     # obj_height = ymax - ymin
        #     # obj_width = xmax - xmin
        #     # new_obj_width = 1.0 * obj_width
        #     # new_obj_height = 1.0 * obj_height
        #
        #     # New_xmin = obj_center_x - new_obj_width/2 if obj_center_x - new_obj_width/2 >= 0 else 0
        #     # New_ymin = obj_center_y - new_obj_height / 2 if obj_center_y - new_obj_height / 2 >= 0 else 0
        #     # New_xmax = obj_center_x + new_obj_width / 2 if obj_center_x + new_obj_width / 2 >= 0 else 0
        #     # New_ymax = obj_center_y + new_obj_height / 2 if obj_center_y + new_obj_height / 2 >= 0 else 0
        #
        #     New_xmin = xmin * 4
        #     New_ymin = ymin * 4
        #     New_xmax = xmax * 4
        #     New_ymax = ymax * 4
        #
        #     obj.find('bndbox').find('xmin').text = str(math.floor(New_xmin))
        #     obj.find('bndbox').find('xmax').text = str(math.ceil(New_xmax))
        #     obj.find('bndbox').find('ymin').text = str(math.floor(New_ymin))
        #     obj.find('bndbox').find('ymax').text = str(math.ceil(New_ymax))
        tree.write(xml_file)


if __name__ == '__main__':
    xml_pic_Resize(xml_path)