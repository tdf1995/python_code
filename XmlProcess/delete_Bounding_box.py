import os
import xml.etree.ElementTree as ET
from os import walk
from xml.dom import minidom
import numpy as np
from xml.dom.minidom import parseString
from lxml.etree import Element, SubElement, tostring

xml_path = r'C:\Users\tdf\Desktop\xml'

def delete_Bounding_box(xml_path):
    for(root, dirnames, filenames)in walk(xml_path):
        for filename in filenames :
            if filename[-4:] == '.xml':
                xml_path = os.path.join(xml_path, filename)
                dom = minidom.parse(xml_path)
                points_b_list = []
                height = dom.getElementsByTagName("height")[0].nodeValue
                width = dom.getElementsByTagName("width")[0].nodeValue
                objs = dom.getElementsByTagName("object")
                label_list = []
                for obj in objs:
                    points_b = []
                    polygen = obj.getElementsByTagName("polygen")
                    pxNode = polygen[0].getElementsByTagName("points_x")
                    pyNode = polygen[0].getElementsByTagName("points_y")
                    px_str = pxNode[0].childNodes[0].nodeValue
                    py_str = pyNode[0].childNodes[0].nodeValue

                    px_ = px_str.split(',')
                    py_ = py_str.split(',')
                    flag_x = []
                    flag_y = []
                    flag_x = [0 if ((int(x) > 0.95 * height) for x in px_) else 1]
                    flag_y = [0 if ((int(y) > 0.95 * width) for y in py_) else 1]
                    if 0 in flag_x or 0 in flag_y:
                        dom.documentElement.removeChild(polygen)

if __name__ == '__main__':
    delete_Bounding_box(xml_path)