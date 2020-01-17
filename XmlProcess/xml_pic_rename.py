# 该脚本用于将图像连同带有目标框信息xml文件一起重命名，除了修改文件名，还要修改xml内的图像路径
import cv2
import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
from os import walk

xml_path = r'E:\多目标跟踪\keras-yolo3-master\keras-yolo3-master\VOCdevkit\VOC2007\新建文件夹'


# def preprocess(root_path):
#    rootdir = os.listdir(root_path)
#    for e in rootdir:
#        subdir = os.path.join(root_path,e)   #   子文件及子文件夹路径
#        if os.path.isfile(subdir):   #   如果是文件
#             if os.path.splitext(subdir)[1] == '.xml':
#                 num = num + 1
#        elif os.path.isdir(subdir):  #   如果是路径
#            preprocess(subdir)

def xml_pic_Rename(xml_path):
    global num
    rootdir = os.listdir(xml_path)
    for e in rootdir:
        subdir = os.path.join(xml_path, e)  # 子文件及子文件夹路径
        if os.path.isfile(subdir):  # 如果是文件
            if os.path.splitext(subdir)[1] == '.xml':
                tree = ET.parse(subdir)
                root = tree.getroot()
                # img_path = root.find('path').text
                img_name = os.path.basename(subdir)[:-4]+'.jpg'
                img_path = os.path.join(xml_path, img_name)
                dirname = os.path.basename(xml_path)
                # if not os.path.exists(img_path):
                    # os.remove(subdir)
                    # continue
                # else:
                # img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

                # new_img_path = os.path.join(xml_path, dirname+'_' + img_name[:-4] + '.jpg')#这里要改
                new_img_path = img_path[:-4]+'_bg.jpg'
                # new_xml_path = os.path.join(xml_path, dirname+'_' + img_name[:-4] + '.xml')
                new_xml_path = subdir[:-4]+'_bg.xml'
                root.find('path').text = new_img_path
                root.find('filename').text = os.path.basename(new_img_path)
                tree.write(new_xml_path)
                os.rename(img_path,new_img_path)
                os.remove(subdir)


        elif os.path.isdir(subdir):  # 如果是路径
            xml_pic_Rename(subdir)




    # xml_files = glob.glob(xml_path+'/*.xml')
    # pic_files = glob.glob(pic_path+'/*.jpg')
    # for pic_file in pic_files:
    #     img = cv2.imdecode(np.fromfile(pic_file, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    #     height, width, channel = img.shape
    #     h_ratio = target_height / height
    #     w_ratio = target_width / width
    #     img = cv2.resize(img, (target_height, target_width), interpolation=cv2.INTER_AREA)
    #     cv2.imencode('.jpg', img)[1].tofile(pic_file)
    #     img_name = os.path.basename(pic_file)
    #     if os.path.join(xml_path, img_name[:-4]+'.xml') in xml_files:
    #         tree = ET.parse(os.path.join(xml_path, img_name[:-4]+'.xml'))
    #         root = tree.getroot()
    #         root.find('size').find('width').text = str(target_width)
    #         root.find('size').find('height').text = str(target_height)
    #         for obj in root.findall('object'):
    #             New_xmin = str(int(int(obj.find('bndbox').find('xmin').text) * target_width / width))
    #             New_xmax = str(int(int(obj.find('bndbox').find('xmax').text) * target_width / width))
    #             New_ymin = str(int(int(obj.find('bndbox').find('ymin').text) * target_height / height))
    #             New_ymax = str(int(int(obj.find('bndbox').find('ymax').text) * target_height / height))
    #
    #             obj.find('bndbox').find('xmin').text = New_xmin
    #             obj.find('bndbox').find('xmax').text = New_xmax
    #             obj.find('bndbox').find('ymin').text = New_ymin
    #             obj.find('bndbox').find('ymax').text = New_ymax
    #         tree.write(os.path.join(xml_path, img_name[:-4]+'.xml'))
if __name__ == '__main__':
    xml_pic_Rename(xml_path)