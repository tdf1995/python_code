# 该脚本用于将图像连同带有目标框信息xml文件一起缩放
import cv2
import os
import glob
import xml.etree.ElementTree as ET
import numpy as np

xml_path = r'E:\多目标跟踪\keras-yolo3-master\keras-yolo3-master\VOCdevkit\VOC2007\新建文件夹'
pic_path = r'E:\多目标跟踪\keras-yolo3-master\keras-yolo3-master\VOCdevkit\VOC2007\新建文件夹'
target_height = 384
target_width = 384

def xml_pic_Resize(xml_path, pic_path):
    xml_files = glob.glob(xml_path+'/*.xml')
    pic_files = glob.glob(pic_path+'/*.jpg')
    for pic_file in pic_files:
        img = cv2.imdecode(np.fromfile(pic_file, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        height, width, channel = img.shape
        h_ratio = target_height / height
        w_ratio = target_width / width
        img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
        cv2.imencode('.jpg', img)[1].tofile(pic_file)
        img_name = os.path.basename(pic_file)
        if os.path.join(xml_path, img_name[:-4]+'.xml') in xml_files:
            tree = ET.parse(os.path.join(xml_path, img_name[:-4]+'.xml'))
            root = tree.getroot()
            root.find('size').find('width').text = str(target_width)
            root.find('size').find('height').text = str(target_height)
            for obj in root.findall('object'):
                New_xmin = str(int(int(obj.find('bndbox').find('xmin').text) * target_width / width))
                New_xmax = str(int(int(obj.find('bndbox').find('xmax').text) * target_width / width))
                New_ymin = str(int(int(obj.find('bndbox').find('ymin').text) * target_height / height))
                New_ymax = str(int(int(obj.find('bndbox').find('ymax').text) * target_height / height))

                obj.find('bndbox').find('xmin').text = New_xmin
                obj.find('bndbox').find('xmax').text = New_xmax
                obj.find('bndbox').find('ymin').text = New_ymin
                obj.find('bndbox').find('ymax').text = New_ymax
            tree.write(os.path.join(xml_path, img_name[:-4]+'.xml'))


        # if pic_file[:-4]+'.xml' in xml_files:
        #     tree = ET.parse(pic_file[:-4]+'.xml')
        #     print(pic_file)
        #     root = tree.getroot()
        #     root.find('size').find('width').text = str(target_width)
        #     root.find('size').find('height').text = str(target_height)
        #     for obj in root.findall('object'):
        #         New_x = ''
        #         New_y = ''
        #         x = np.array(
        #             [(float(str(c))* w_ratio) for c in list(obj.find('polygen').find("points_x").text.strip().split(",")[:-1])])
        #         y = np.array(
        #             [(float(str(c))* h_ratio) for c in list(obj.find('polygen').find("points_y").text.strip().split(",")[:-1])])
        #         for i in range(len(x)):
        #             New_x = New_x + str(x[i]) + ','
        #         obj.find('polygen').find("points_x").text = New_x
        #         for i in range(len(y)):
        #             New_y = New_y + str(y[i]) + ','
        #         obj.find('polygen').find("points_y").text = New_y
        #     tree.write(pic_file[:-4]+'.xml')
if __name__ == '__main__':
    xml_pic_Resize(xml_path, pic_path)