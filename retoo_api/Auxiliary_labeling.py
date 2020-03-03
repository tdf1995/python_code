import os
import glob
import xml.etree.ElementTree as ET
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString

def name2index(dir_path):
    '''
    用于生成分类类别名与对应的label的函数，写在.txt文件中
    :param dir_path: 分类数据根目录
    :return: 对应字典
    '''
    names = os.listdir(dir_path)
    f = open(os.path.join(dir_path,os.path.basename(dir_path)+'.txt'),'w')
    dict = {}
    for index, name in enumerate(names):
        f.write(name+':'+str(index)+'\n')
        dict[index] = name
        # os.rename(os.path.join(dir_path,name),os.path.join(dir_path,str(index)))
    f.close()
    return dict

def xml_polygen2bbox(xml_path, new_path=None):
    '''
    将多边形标注转化为矩形框标注保存
    :param xml_path: xml路径
    :return: 空
    '''
    tree = ET.parse(xml_path)
    # annotation = tree.find('annotation')
    width = tree.find('size').find('width').text
    height = tree.find('size').find('height').text
    path = tree.find('path').text
    filename = tree.find('filename').text

    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'VOC'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = os.path.basename(path)

    node_path = SubElement(node_root, 'path')
    node_path.text = path

    node_source = SubElement(node_root, 'source')
    node_database = SubElement(node_source, 'database')
    node_database.text = 'tdf'

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = width
    node_height = SubElement(node_size, 'height')
    node_height.text = height
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'


    objs = tree.findall('object')
    for obj in objs:
        name = obj.find('name').text
        polygen = obj.find('polygen')
        points_x = [int(float(i)) for i in polygen.find('points_x').text.split(',')]
        (xmin, xmax) = (min(points_x), max(points_x))
        points_y = [int(float(i)) for i in polygen.find('points_y').text.split(',')]
        (ymin, ymax) = (min(points_y), max(points_y))

        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = name
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'

        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(xmin)
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(ymin)
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(xmax)
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(ymax)

    xml = tostring(node_root, pretty_print= True)
    dom = parseString(xml)
    if new_path:
        with open(new_path, 'wb') as f:
            f.write(dom.toprettyxml(indent='\t', encoding='utf-8'))
    else:
        with open(xml_path, 'wb') as f:
            f.write(dom.toprettyxml(indent='\t', encoding='utf-8'))

def xml_ChangeLabel(xml_path, oldLabel, newLabel, new_path=None):
    '''
    修改xml某个类别的label
    :param xml_path:xml文件路径
    :param oldLabel:待修改label
    :param newLabel:新label
    :param new_path: 保存路径
    :return:
    '''
    tree = ET.parse(xml_path)
    objs = tree.findall('object')
    for obj in objs:
        name = obj.find('name').text
        if name == oldLabel:
            obj.find('name').text = newLabel
        else:
            continue
    if new_path:
        tree.write(new_path)
    else:
        tree.write(xml_path)

def xml_DeleteLabel(xml_path, label, new_path=None):
    '''
    删除xml文件某个类别的标注框
    :param xml_path: xml文件路径
    :param label: 待删除label
    :param new_path: 保存路径
    :return:
    '''
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objs = tree.findall('object')
    for i,obj in enumerate(objs):
        name = obj.find('name').text
        if name == label:
            root.remove(obj)
        else:
            continue
    if new_path:
        tree.write(new_path)
    else:
        tree.write(xml_path)

def xml_ResizeLabel(xml_path, offset=[], p_x=None,p_y =None,new_path =None):
    '''
    对标注框进行大小变换，像素或比例
    :param xml_path: xml路径
    :param offset: （xmin,ymin,xmax,ymax）的变化像素
    :param p_x: x比例变化
    :param p_y: y比例变化
    :param new_path: 保存路径
    :return:
    '''
    if offset is not []:
        try:
            [xmin,ymin,xmax,ymax] = offset
            tree = ET.parse(xml_path)
            objs = tree.findall('object')
            for obj in objs:
                bbox = obj.find('bndbox')
                bbox.find('xmin').text = int(bbox.find('xmin').text) + xmin
                bbox.find('xmax').text = int(bbox.find('xmax').text) + xmax
                bbox.find('ymin').text = int(bbox.find('ymin').text) + ymin
                bbox.find('ymax').text = int(bbox.find('ymax').text) + ymax
            if new_path:
                tree.write(new_path)
            else:
                tree.write(xml_path)
        except:
            print('error:offset must be 4 numbers')
    elif p_x is not None and p_y is not None:
        tree = ET.parse(xml_path)
        objs = tree.findall('object')
        for obj in objs:
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            center_x = (xmin + xmax)/2
            center_y = (ymin+ymax)/2
            new_xmin = center_x - (xmax-xmin)/2*p_x
            new_xmax = center_x + (xmax - xmin) / 2 * p_x
            new_ymin = center_y - (ymax - ymin) / 2 * p_y
            new_ymax = center_y + (ymax - ymin) / 2 * p_y
            bbox.find('xmin').text = int(new_xmin)
            bbox.find('xmax').text = int(new_xmax)
            bbox.find('ymin').text = int(new_ymin)
            bbox.find('ymax').text = int(new_ymax)
        if new_path:
            tree.write(new_path)
        else:
            tree.write(xml_path)
    else:
        print('you should set some parameters')




def ocr_merge_bbox(xml_path, new_path=None, class_num=1):
    '''
    ocr项目，将所有标注框合并成一个标注框
    :param xml_path: xml路径
    :param new_path: 保存路径
    :return:
    '''

    tree = ET.parse(xml_path)
    objs = tree.findall('object')
    xmin = 10000
    ymin = 10000
    xmax = 0
    ymax = 0
    for i,obj in enumerate(objs):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        xmax = int(bbox.find('xmax').text)
        ymin = int(bbox.find('ymin').text)
        ymax = int(bbox.find('ymax').text)
        center_y = (ymin +ymax)/2


def detXML_delete(xml_path, Attribute, flag,save_path=None):
    '''
    按标注框的属性(大小，长宽，长宽比，边距等)删除xml文件中的某些特殊标注框
    :param xml_path:xml路径
    :param Attribute:'height','width','ratio','interval'
    :param flag:'<>num'
    :param save_path:xml保存路径
    :return:
    '''
    tree = ET.parse(xml_path)
    root = tree.getroot()
    Width = int(tree.find('size').find('width').text)
    Height = int(tree.find('size').find('height').text)
    objs = tree.findall('object')
    flag1 = flag[0]
    num = int(flag[1:])
    for obj in objs:
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        xmax = int(bbox.find('xmax').text)
        ymin = int(bbox.find('ymin').text)
        ymax = int(bbox.find('ymax').text)
        height = xmax - xmin
        width = ymax - ymin
        ratio = width/height
        center_x = (xmax+xmin)/2
        center_y = (ymax+ymin)/2
        interval = min((Height-height),(Width-width))
        if Attribute=='height':
            if flag1=='>'and height>num:
                root.remove(obj)
            elif flag1=='<'and height<num:
                root.remove(obj)
            else:
                continue
        elif Attribute=='width':
            if flag1=='>'and width>num:
                root.remove(obj)
            elif flag1=='<'and width<num:
                root.remove(obj)
            else:
                continue
        elif Attribute=='ratio':
            if flag1=='>'and ratio>num:
                root.remove(obj)
            elif flag1=='<'and ratio<num:
                root.remove(obj)
            else:
                continue
        elif Attribute=='interval':
            if flag1=='>'and interval>num:
                root.remove(obj)
            elif flag1=='<'and interval<num:
                root.remove(obj)
            else:
                continue
        else:
            print('请输入正确的属性')
    if save_path:
        tree.write(save_path)
    else:
        tree.write(xml_path)
if __name__=="__main__":
    name2index('E:\ocr\单字符识别\clean\clean')