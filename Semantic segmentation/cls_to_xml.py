# 根据语义分割预测label图生成xml
import cv2
import os
from os import walk
import numpy as np
from xml.dom import minidom
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
from lxml.etree import Element, SubElement, tostring
import collections


root_path = r'E:\ocr\dataset'

def gen_xml_from_predict(root_path):
    for(root_path, dirnames, _)in walk(root_path):
        for dirname in dirnames:
            sub_path = os.path.join(root_path, dirname)
            for(_, _, filenames)in walk(sub_path):
                for filename in filenames:
                    if filename[-4:] =='.png':
                        pic_path = os.path.join(sub_path, filename)
                        print(pic_path)
                        mask = cv2.imdecode(np.fromfile(pic_path, dtype=np.uint8), -1)
                        h = mask.shape[0]
                        w = mask.shape[1]


                        _, mask_Bin = cv2.threshold(mask,0,255,cv2.THRESH_BINARY)
                        contours, hierarchy = cv2.findContours(mask_Bin,
                                                      cv2.RETR_EXTERNAL,
                                                      cv2.CHAIN_APPROX_NONE)

                        node_root = Element('annotation')
                        node_folder = SubElement(node_root, 'folder')
                        node_folder.text = dirname

                        node_filename = SubElement(node_root, 'filename')
                        node_filename.text = filename[:-4] + '.jpg'

                        node_path = SubElement(node_root, 'path')
                        node_path.text = pic_path

                        node_source = SubElement(node_root, 'source')
                        node_database = SubElement(node_source, 'database')
                        node_database.text = 'tdf'

                        node_size = SubElement(node_root, 'size')
                        node_width = SubElement(node_size, 'width')
                        node_width.text = str(w)
                        node_height = SubElement(node_size, 'height')
                        node_height.text = str(h)
                        node_depth = SubElement(node_size, 'depth')
                        node_depth.text = '3'

                        node_segmented = SubElement(node_root, 'segmented')
                        node_segmented.text = '0'



                        if len(contours) > 0:
                            # max_cont_index = 0
                            # max_cont = 0
                            contours_OK = []
                            max_thresh = 0#800
                            for j in range(len(contours)):
                                arclen = cv2.arcLength(contours[j], True)
                                if arclen > max_thresh:
                                    contours_OK.append(contours[j])
                                    # max_cont = arclen
                                    # max_cont_index = j
                            # contours = contours.tolist()
                            # 得到最大轮廓
                            # max_length_cont = contours[max_cont_index]
                            contours = contours_OK
                            for h in range(len(contours)):
                            # 多边形近似
                                approx = cv2.approxPolyDP(contours[h], 0.6, True)
                                pt_x = ""
                                pt_y = ""
                                for k in range(len(approx)):
                                    # if k%8==0: #或者平均间隔
                                    pt = approx[k]
                                    pt_x += str(pt[0][0])
                                    pt_y += str(pt[0][1])

                                    pt_x += ","
                                    pt_y += ","

                                node_object = SubElement(node_root, 'object')
                                node_name = SubElement(node_object, 'name')
                                # node_name.text = dirname.split('.', 1)[0]
                                # node_name.text = str(cls)
                                node_name.text = '1'

                                node_pose = SubElement(node_object, 'pose')
                                node_pose.text = 'Unspecified'
                                node_truncated = SubElement(node_object, 'truncated')
                                node_truncated.text = '0'
                                node_polygen = SubElement(node_object, 'polygen')
                                node_points_x = SubElement(node_polygen, 'points_x')
                                node_points_x.text = str(pt_x)
                                node_points_y = SubElement(node_polygen, 'points_y')
                                node_points_y.text = str(pt_y)

                            xml = tostring(node_root, pretty_print= True)
                            dom = parseString(xml)

                            xml_path = os.path.join(pic_path[:-4] + '.xml')
                            with open(xml_path, 'wb') as f:
                                f.write(dom.toprettyxml(indent='\t', encoding='utf-8'))
if __name__ == '__main__':
    gen_xml_from_predict(root_path)