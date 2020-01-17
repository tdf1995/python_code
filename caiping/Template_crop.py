import os
import glob
import cv2
import xml.etree.ElementTree as ET
import numpy as np

Template_xml = r'E:\菜品\分割\反\2019.12.12 18.03.14.271_seg_loc.xml'
pic_path = r'E:\菜品\分割\反'
target_path = r'E:\菜品\分割\1111_crop'

def read_template(root,img,img_name):
    i = 0
    for obj in root.findall('object'):

        New_xmin = int(int(obj.find('bndbox').find('xmin').text) )
        New_xmax = int(int(obj.find('bndbox').find('xmax').text) )
        New_ymin = int(int(obj.find('bndbox').find('ymin').text) )
        New_ymax = int(int(obj.find('bndbox').find('ymax').text) )
        img_crop = img[New_ymin:New_ymax, New_xmin:New_xmax]
        cv2.imencode('.jpg',img_crop)[1].tofile(os.path.join(target_path,img_name[:-4]+'_'+str(i)+'.jpg'))
        i = i+1
tree = ET.parse(Template_xml)
print('read xml correct!')
root = tree.getroot()
pic_files = glob.glob(pic_path+'/*.jpg')
for pic_file in pic_files:
    img = cv2.imdecode(np.fromfile(pic_file, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    read_template(root,img,os.path.basename(pic_file))