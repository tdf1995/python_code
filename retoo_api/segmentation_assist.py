import cv2
import  os
from xml.dom import minidom
from os import walk
import shutil
import numpy as np
import os
import glob
import xml.etree.ElementTree as ET
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import re

def segXml2segPic(xml_path, mode='semantic',save_path=None):
    '''
    根据多边形标注生成语义分割、实例分割的二值图
    :param xml_path:xml路径
    :param mode:选择'semantic'，'instance'
    :param save_path:图像保存的文件夹路径
    :return:
    '''
    print(xml_path)
    tree = ET.parse(xml_path)
    objs = tree.findall('object')
    image = cv2.imdecode(np.fromfile(xml_path[:-4]+'.jpg',dtype=np.uint8),-1)
    height, width = image.shape
    filename = tree.find('filename').text
    if mode == 'semantic':
        Cls_Pic = np.zeros([height, width], dtype=np.uint8)
        for obj in objs:
            points_x = obj.find('polygen').find('points_x').text.split(',')
            points_y = obj.find('polygen').find('points_y').text.split(',')
            if points_x[-1] == '':
                point_num = len(points_x)-1
            else:
                point_num = len(points_x)
            ps = np.zeros([point_num, 2], dtype=np.int32)
            for i in range(point_num):
                if points_x[i] == '':
                    continue
                ps[i] = (int(float(points_x[i])), int(float(points_y[i])))
            cv2.polylines(Cls_Pic, [ps], 1, 0, 1)  # img:图像,顶点集，是否闭合，颜色，线宽度
            cv2.fillPoly(Cls_Pic, [ps], 255)
        Cls_Pic_name = filename[:-4]+'.png'
        if save_path:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            cv2.imencode('.jpg',Cls_Pic)[1].tofile(os.path.join(save_path,os.path.basename(xml_path)[:-4]+'.png'))
        else:
            cv2.imencode('.png', Cls_Pic)[1].tofile(xml_path[:-4]+'.png')
    elif mode=='instance':
        Ins_Pic = np.zeros([width, height], dtype=np.uint16)
        for i,obj in enumerate(objs):
            points_x = obj.find('polygen').find('points_x').text.split(',')[:-1]
            points_y = obj.find('polygen').find('points_y').text.split(',')[:-1]
            point_num = len(points_x)
            ps = np.zeros([point_num, 2], dtype=np.int32)
            for i in range(point_num-1):
                ps[i] = (int(float(points_x[i])), int(float(points_y[i])))
                cv2.polylines(Ins_Pic, [ps], 1, 0, 1)  # img:图像,顶点集，是否闭合，颜色，线宽度
                cv2.fillPoly(Ins_Pic, [ps], i)
        Cls_Pic_name = filename[:-4]+'.png'
        if save_path:
            cv2.imencode('.png',Ins_Pic)[1].tofile(os.path.join(save_path, Cls_Pic_name))
        else:
            cv2.imencode('.png', Ins_Pic)[1].tofile(os.path.join(os.path.dirname(xml_path), Cls_Pic_name))

def segXml2cropPic(xml_path, mode='semantic',save_path=None):
    '''
    根据多边形标注裁剪最小外接矩形区域
    :param xml_path:xml路径
    :param mode:选择'semantic'，'instance'
    :param save_path:图像保存的文件夹路径
    :return:
    '''
    print(xml_path)
    tree = ET.parse(xml_path)
    objs = tree.findall('object')
    image = cv2.imdecode(np.fromfile(xml_path[:-4]+'.jpg',dtype=np.uint8),-1)
    height, width = image.shape
    filename = tree.find('filename').text
    Cls_Pic = np.zeros([height, width], dtype=np.uint8)
    for obj in objs:
        points_x = obj.find('polygen').find('points_x').text.split(',')
        points_y = obj.find('polygen').find('points_y').text.split(',')
        if points_x[-1] == '':
            point_num = len(points_x)-1
        else:
            point_num = len(points_x)
        ps = np.zeros([point_num, 2], dtype=np.int32)
        for i in range(point_num):
            if points_x[i] == '':
                continue
            # elif float(points_x[i]) > width:
            #     points_x[i] = width
            # if float(points_y[i]) > height:
            #     points_y[i] = height
            ps[i] = (int(float(points_x[i])), int(float(points_y[i])))
        rotated_rect = cv2.minAreaRect(ps)
        # rotated_rect[1][0] = 1.1*rotated_rect[1][0]
        # rotated_rect[1][1] = 1.1*rotated_rect[1][1]
        rect_max = int(1.1*rotated_rect[1][0]) if rotated_rect[1][0] > rotated_rect[1][1] else int(1.1*rotated_rect[1][1])
        rect_min = int(1.1*rotated_rect[1][1]) if rotated_rect[1][0] > rotated_rect[1][1] else int(1.1*rotated_rect[1][0])
        if rect_max/2+int(rotated_rect[0][0])>width:
            rect_max = (width - int(rotated_rect[0][0]))*2
        # if rect_max / 2 + int(rotated_rect[0][0]) > width:

        # if rect_min<min(height,width):
        #     rect_min = min(height, width)

        angle = rotated_rect[2] if abs(rotated_rect[2]) < 45 else (rotated_rect[2] + 90)
        M = cv2.getRotationMatrix2D((int(rotated_rect[0][0]), int(rotated_rect[0][1])), angle, 1)
        rotated_img = cv2.warpAffine(image, M, (width, height))
        rotated_img = cv2.getRectSubPix(rotated_img, (rect_max, rect_min),
                                        (int(rotated_rect[0][0]), int(rotated_rect[0][1])))
        if save_path:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            cv2.imencode('.jpg',rotated_img)[1].tofile(os.path.join(save_path,os.path.basename(xml_path)[:-4]+'_crop.jpg'))
        else:
            cv2.imencode('.jpg', rotated_img)[1].tofile(xml_path[:-4]+'_crop.jpg')

def bboxXml2cropPic(xml_path,save_path,label_path=None):
    '''
    根据多边形标注裁剪最小外接矩形区域
    :param xml_path:xml路径
    :param mode:选择'semantic'，'instance'
    :param save_path:图像保存的文件夹路径
    :param label_path:label对应dict的txt路径
    :return:
    '''
    # print(xml_path)
    tree = ET.parse(xml_path)
    objs = tree.findall('object')
    if not os.path.exists(xml_path[:-4]+'.jpg'):
        return 0
    image = cv2.imdecode(np.fromfile(xml_path[:-4]+'.jpg',dtype=np.uint8),-1)
    height, width = image.shape
    # filename = tree.find('filename').text
    # Cls_Pic = np.zeros([height, width], dtype=np.uint8)
    if not label_path:
        for i, obj in enumerate(objs):
            xmin = int(obj.find('bndbox').find('xmin').text)
            ymin = int(obj.find('bndbox').find('ymin').text)
            xmax = int(obj.find('bndbox').find('xmax').text)
            ymax = int(obj.find('bndbox').find('ymax').text)
            crop_img = image[ymin:ymax, xmin:xmax]
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            print(os.path.join(save_path, (os.path.basename(xml_path)[:-4] + '_' + str(i) + '_crop.jpg')))
            try:
                cv2.imencode('.jpg', crop_img)[1].tofile(
                    os.path.join(save_path, (os.path.basename(xml_path)[:-4] + '_' + str(i) + '_crop.jpg')))
            except:
                continue
    else:
        f = open(label_path,'r')
        lines = f.readlines()
        b={}
        for i,line in enumerate(lines):
            a = re.split('[::\n]',line)
            if len(a)>3:
                b[':'] = a[2]
            if a[0]=='':
                continue
            b[a[0]] = a[1]

        for i,obj in enumerate(objs):
            label = obj.find('name').text
            if label in b:
                label = b[label]
            print(label)
            xmin = int(obj.find('bndbox').find('xmin').text)
            ymin =int(obj.find('bndbox').find('ymin').text)
            xmax = int(obj.find('bndbox').find('xmax').text)
            ymax = int(obj.find('bndbox').find('ymax').text)
            crop_img = image[ymin:ymax, xmin:xmax]

            sub_dir = os.path.join(save_path, label)
            if not os.path.exists(sub_dir):
                os.mkdir(sub_dir)
            try:
                cv2.imencode('.jpg',crop_img)[1].tofile(os.path.join(sub_dir,os.path.basename(xml_path)[:-4]+'_'+str(i)+'_crop.jpg'))
            except:
                continue

def bboxXml2segPic(xml_path,save_path=None):
    '''
    根据多边形标注生成语义分割、实例分割的二值图
    :param xml_path:xml路径
    :param mode:选择'semantic'，'instance'
    :param save_path:图像保存的文件夹路径
    :return:
    '''
    print(xml_path)
    tree = ET.parse(xml_path)
    objs = tree.findall('object')
    image = cv2.imdecode(np.fromfile(xml_path[:-4]+'.jpg',dtype=np.uint8),-1)
    (height, width,_) = image.shape
    filename = tree.find('filename').text

    Cls_Pic = np.zeros([height, width], dtype=np.uint8)
    for obj in objs:
        xmin = obj.find('bndbox').find('xmin').text
        ymin = obj.find('bndbox').find('ymin').text
        xmax = obj.find('bndbox').find('xmax').text
        ymax = obj.find('bndbox').find('ymax').text
        ps = np.zeros([4, 2], dtype=np.int32)

        ps[0] = (int(xmin), int(ymin))
        ps[3] = (int(xmax), int(ymin))
        ps[1] = (int(xmin), int(ymax))
        ps[2] = (int(xmax), int(ymax))
        cv2.polylines(Cls_Pic, [ps], 1, 0, 1)  # img:图像,顶点集，是否闭合，颜色，线宽度
        cv2.fillPoly(Cls_Pic, [ps], 1)
        # cv2.imshow('1',Cls_Pic)
        # cv2.waitKey(0)
    Cls_Pic_name = filename[:-4]+'.png'
    if save_path:

        cv2.imencode('.png',Cls_Pic)[1].tofile(os.path.join(save_path, Cls_Pic_name))
    else:
        cv2.imencode('.png', Cls_Pic)[1].tofile(os.path.join(os.path.dirname(xml_path), Cls_Pic_name))

def segPic2xml(pic_path, mode='semantic',save_path=None):
    '''
    根据有分割信息的灰度图生成多边形标注
    :param pic_path: 灰度图位置
    :param mode: 'semantic'或者'instance'
    :param save_path: xml保存路径
    :return:
    '''
    mask = cv2.imdecode(np.fromfile(pic_path, dtype=np.uint8), -1)
    h = mask.shape[0]
    w = mask.shape[1]
    if mode=='semantic':
        _, mask_Bin = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(mask_Bin,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)

        node_root = Element('annotation')
        node_folder = SubElement(node_root, 'folder')
        node_folder.text = os.path.basename(os.path.dirname(pic_path))

        node_filename = SubElement(node_root, 'filename')
        node_filename.text = os.path.basename(pic_path)

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

            xml = tostring(node_root, pretty_print=True)
            dom = parseString(xml)
            if not save_path:
                xml_path = os.path.join(pic_path[:-4] + '.xml')
            else:
                xml_path = os.path.join(save_path. os.path.basename(pic_path)[:-4]+'.xml')
            with open(xml_path, 'wb') as f:
                f.write(dom.toprettyxml(indent='\t', encoding='utf-8'))
    elif mode=='instance':
        instance_num = max(mask)

        node_root = Element('annotation')
        node_folder = SubElement(node_root, 'folder')
        node_folder.text = os.path.basename(os.path.dirname(pic_path))

        node_filename = SubElement(node_root, 'filename')
        node_filename.text = os.path.basename(pic_path)

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
        for i in range(instance_num):
            _, mask_Bin = cv2.threshold(mask, i, 0, cv2.THRESH_BINARY)# 大于i的置为0
            mask = mask - mask_Bin # 去除已经处理的实例

            contours, hierarchy = cv2.findContours(mask_Bin,
                                                   cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_NONE)
            if len(contours) > 0:
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
                    node_name.text = str(i)

                    node_pose = SubElement(node_object, 'pose')
                    node_pose.text = 'Unspecified'
                    node_truncated = SubElement(node_object, 'truncated')
                    node_truncated.text = '0'
                    node_polygen = SubElement(node_object, 'polygen')
                    node_points_x = SubElement(node_polygen, 'points_x')
                    node_points_x.text = str(pt_x)
                    node_points_y = SubElement(node_polygen, 'points_y')
                    node_points_y.text = str(pt_y)

            xml = tostring(node_root, pretty_print=True)
            dom = parseString(xml)
            if not save_path:
                xml_path = os.path.join(pic_path[:-4] + '.xml')
            else:
                xml_path = os.path.join(save_path.os.path.basename(pic_path)[:-4] + '.xml')
            with open(xml_path, 'wb') as f:
                f.write(dom.toprettyxml(indent='\t', encoding='utf-8'))

def segXML_delete(xml_path, Attribute, flag,save_path=None):
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
        polygen = obj.find('polygen')
        points_x = [int(float(i)) for i in polygen.find('points_x').text.split(',')]
        (xmin, xmax) = (min(points_x), max(points_x))
        height = xmax - xmin
        points_y = [int(float(i)) for i in polygen.find('points_y').text.split(',')]
        (ymin, ymax) = (min(points_y), max(points_y))
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


def segMask_Crop(image, mask, area_thresh=3500):
    '''
    根据原图和掩码图，得到最小外接矩形同时返回掩码区图像
    :param image:原图
    :param mask:掩码图
    :param area_thresh:面积阈值，小于此阈值的掩码区域忽略
    :return:掩码区图像和最小外接矩形
    '''
    height = image.shape[0]
    width = image.shape[1]
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area<area_thresh:
            continue
        rotated_rect = cv2.minAreaRect(contours[i])  # 0为中心点坐标，1为宽高
        box = cv2.boxPoints(rotated_rect)

        box = np.int0(box)
        rect_max = int(rotated_rect[1][0]) if rotated_rect[1][0] > rotated_rect[1][1] else int(rotated_rect[1][1])
        rect_min = int(rotated_rect[1][1]) if rotated_rect[1][0] > rotated_rect[1][1] else int(rotated_rect[1][0])

        angle = rotated_rect[2] if abs(rotated_rect[2]) < 45 else (rotated_rect[2] + 90)
        M = cv2.getRotationMatrix2D((int(rotated_rect[0][0]), int(rotated_rect[0][1])), angle, 1)
        rotated_img = cv2.warpAffine(image, M, (width, height))
        rotated_img = cv2.getRectSubPix(rotated_img, (rect_max, rect_min),
                                        (int(rotated_rect[0][0]), int(rotated_rect[0][1])))
        return  rotated_img,rotated_rect

if __name__=="__main__":
    path = r'\\192.168.1.251\ssd-研发部\项目工作目录\OCR项目\拍摄的数据集\字符区图片数据集\邦纳提供图片_字符区图_单字符矩形标注'
    save_path = r'\\192.168.1.251\ssd-研发部\项目工作目录\OCR项目\拍摄的数据集\字符区图片数据集\邦纳提供图片_字符区图_单字符矩形标注_crop'
    # label_path = r'D:\python code\ocr_dict.txt'
    xml_files = glob.glob(path+'\*\*.xml')
    for xml_file in xml_files:
        dir = xml_file.split('\\')
        Save_path = os.path.join(save_path,dir[-3],dir[-2])
        bboxXml2cropPic(xml_file,Save_path)