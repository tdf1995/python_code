import glob
import xml.etree.ElementTree as ET
import cv2
import os
import numpy as np
root_path = r'E:\多目标跟踪\Market-1501-v15.09.15\VOCdevkit\VOC2007\已标注\Annotations'
crop_path = r'E:\多目标跟踪\Market-1501-v15.09.15\VOCdevkit\VOC2007\crop'
xml_files = glob.glob(root_path+'/*.xml')
for xml_file in xml_files:
    tree = ET.parse(xml_file)
    pic_path = xml_file[:-4]+'.jpg'
    pic_name = os.path.basename(pic_path)[:-4]
    img = cv2.imdecode(np.fromfile(pic_path,dtype=np.uint8),-1)
    objs = tree.findall('object')
    for obj in objs:
        xmin = int(obj.find('bndbox').find('xmin').text)
        xmax = int(obj.find('bndbox').find('xmax').text)
        ymin = int(obj.find('bndbox').find('ymin').text)
        ymax = int(obj.find('bndbox').find('ymax').text)
        id = obj.find('name').text
        crop_img = img[ymin:ymax, xmin:xmax]
        crop_img_name = id.zfill(4)+'C1T0001F'+pic_name[-4:]+'.jpg'
        if not os.path.exists(os.path.join(crop_path, id.zfill(4))):
            os.mkdir(os.path.join(crop_path, id.zfill(4)))
        cv2.imencode('.jpg',crop_img)[1].tofile(os.path.join(crop_path, id.zfill(4),crop_img_name))
        # cv2.imshow('1',crop_img)
        # cv2.waitKey(0)
        # print(1)