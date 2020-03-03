import os
import glob
import numpy as np
import cv2
from xml.dom.minidom import parseString
from lxml.etree import Element, SubElement, tostring

pic_root_path = r'C:\Users\tdf\Desktop\WiderPerson\Images'
txt_root_path = r'C:\Users\tdf\Desktop\WiderPerson\Annotations'

txt_files = glob.glob(txt_root_path+'/*.txt')
for txt_file in txt_files:
    pic_path = txt_file[:-4]
    img = cv2.imdecode(np.fromfile(pic_path, dtype=np.uint8), -1)
    (height, width, _) = img.shape
    f = open(txt_file,'rb')
    lines = f.readlines()

    # 开始写xml
    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'VOC'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = os.path.basename(pic_path)

    node_path = SubElement(node_root, 'path')
    node_path.text = pic_path

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


    for line in lines[1:]:
        b = line.decode().split(' ')
        print(1)
        (xmin, ymin, xmax, ymax) = (b[1],b[2],b[3],b[4])
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = '1'
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'

        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(int(xmin))
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(int(ymin))
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(int(xmax))
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(int(ymax))

    xml = tostring(node_root, pretty_print=True)
    dom = parseString(xml)
    with open(pic_path[:-4]+'.xml', 'wb') as f:
        f.write(dom.toprettyxml(indent='\t', encoding='utf-8'))