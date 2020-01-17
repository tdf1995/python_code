# 这个程序用于检查xml文件有误各种错误
import os
from os import walk
from xml.dom import minidom
import xml.etree.ElementTree as ET

xml_path = r'F:\models-master\cell\VOCdevkit\VOC2007\Annotations'
# 判断坐标有没有超出边界，或交叉
def IsCoordinateRight(dom):
    filename = dom.getElementsByTagName("filename")
    filename = filename[0].childNodes[0].nodeValue
    nWidth = dom.getElementsByTagName("width")
    width = int(nWidth[0].childNodes[0].nodeValue)
    nHeight = dom.getElementsByTagName("height")
    height = int(nHeight[0].childNodes[0].nodeValue)

    objs = dom.getElementsByTagName("object")
    for i in range(objs.length):
        xmin = objs[i].getElementsByTagName("xmin")
        xmin = int(xmin[0].childNodes[0].nodeValue)
        xmax = objs[i].getElementsByTagName("xmax")
        xmax = int(xmax[0].childNodes[0].nodeValue)
        ymin = objs[i].getElementsByTagName("ymin")
        ymin = int(ymin[0].childNodes[0].nodeValue)
        ymax = objs[i].getElementsByTagName("ymax")
        ymax = int(ymax[0].childNodes[0].nodeValue)
        if (xmin > width and xmax > width) or xmin > xmax:
            print("there are some error in " + filename)
        if (ymin > height and ymax > height) or ymin > ymax:
            print("there are some error in " + filename)

def IsfoldRight(xmlFileName, root):
    if not root.find('folder').text == 'VOC2007':
        root.find('folder').text = 'VOC2007'
    objs = root.findall('object')
    for obj in objs:
        if not obj.find('name').text == 'danhe':
            obj.find('name').text = 'danhe'
    tree.write(xmlFileName)

if __name__ == "__main__":
    file = []
    for (dirpath, dirnames, filenames) in walk(xml_path):
        file.extend(filenames)
        break
    for xmlFileName in file:
        if(xmlFileName [-4:] != ".xml"):
            continue
        tree = ET.parse(os.path.join(xml_path, xmlFileName))
        root = tree.getroot()
        # IsCoordinateRight(dom)
        IsfoldRight(os.path.join(xml_path, xmlFileName), root)