# 把crowdhuman数据集的标注从odgt转为voc格式
import os
import re
import json
from xml.dom.minidom import parseString
from lxml.etree import Element, SubElement, tostring
import numpy as np
import glob
import cv2
def load_file(fpath):#fpath是具体的文件 ，作用：#str to list
    assert os.path.exists(fpath)  #assert() raise-if-not
    with open(fpath,'r') as fid:
        lines = fid.readlines()
    records = [json.loads(line.strip('\n')) for line in lines] #str to list
    return records

def Write_Detection(image, boxes, image_path):
    min_score_thresh = 0.4
    (height, width, _) = image.shape
    box_num, _ = np.shape(boxes)


    # 开始写xml
    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'VOC'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = os.path.basename(image_path)

    node_path = SubElement(node_root, 'path')
    node_path.text = image_path

    node_source = SubElement(node_root, 'source')
    node_database = SubElement(node_source, 'database')
    node_database.text = 'tdf'

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(width)
    node_height = SubElement(node_size, 'height')
    node_height.text = str(height)
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    for j in range(len(boxes)):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = '1'
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'

        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(int(boxes[j][1]))
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(int(boxes[j][0]))
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(int(boxes[j][3]+boxes[j][1]))
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(int(boxes[j][2]+boxes[j][0]))
    xml = tostring(node_root, pretty_print= True)
    dom = parseString(xml)

    return dom

odgt_path = r'E:\多目标跟踪\annotation_train.odgt'
target_path = r'E:\多目标跟踪\crowdhuman'
image_path = r'E:\多目标跟踪\CrowdHuman_train01\Images'
records = load_file(odgt_path)
pic_files = glob.glob(image_path+'/*.jpg')
for record in records:
    pic_name = record['ID']+'.jpg'
    gtboxes = record['gtboxes']
    boxes = []
    for gtbox in gtboxes:
        fbox = gtbox['fbox']
        boxes.append(fbox)
    img_path = os.path.join(image_path,pic_name)
    xml_path = img_path[:-4]+'.xml'
    if img_path in pic_files:
        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        dom = Write_Detection(image, boxes, img_path)
        with open(xml_path, 'wb') as f:
            f.write(dom.toprettyxml(indent='\t', encoding='utf-8'))
