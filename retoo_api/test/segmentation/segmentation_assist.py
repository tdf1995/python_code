import cv2
import numpy as np
import xml.etree.ElementTree as ET

def simple_show_mask(mask):
    '''
    简单显示mask
    :param mask:
    :return:
    '''
    _, mask_binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    return mask_binary

def draw_mask_in_img(image, mask, draw_boundingRect=True,
                draw_minAreaRect=True,draw_contour=True):
    '''
    在原图上画mask的各种图形
    :param image: 原图
    :param mask: 预测图
    :param draw_boundingRect: 是否画bbox
    :param draw_minAreaRect: 是否画最小外接矩形
    :param draw_contour: 是否画轮廓
    :return: 画好的图
    '''
    height = image.shape[0]
    width = image.shape[1]
    mask = cv2.resize(mask, (width,height),cv2.INTER_NEAREST)
    _, mask_binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    contours,hierarchy= cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours)>0:
        for i in range(len(contours)):
            if draw_boundingRect:
                x, y, w, h = cv2.boundingRect(contours[i])
                xmin, ymin, xmax, ymax = x, y, x + w, y + h
                cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(255,0,0),5)
            if draw_minAreaRect:
                rotated_rect = cv2.minAreaRect(contours[i])
                box = cv2.boxPoints(rotated_rect)
                box = np.int0(box)
                cv2.drawContours(image, [box], 0, (0, 255, 0), 5)
            if draw_contour:
                cv2.drawContours(image, contours, i, (0,0,255),5)
    return image

def crop_imgs(image, mask):
    '''
    根据预测，将原图像中的分割物体割出来
    :param image: 原图
    :param mask: 预测图
    :return: 分割后的小图
    '''
    height = image.shape[0]
    width = image.shape[1]
    mask = cv2.resize(mask, (width, height), cv2.INTER_NEAREST)
    _, mask_binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    num_contours = len(contours)
    rotated_img=[num_contours]
    for i in range(num_contours):
        rotated_rect = cv2.minAreaRect(contours[i])# 0为中心点坐标，1为宽高

        rect_max = int(rotated_rect[1][0]) if rotated_rect[1][0] > rotated_rect[1][1] else int(rotated_rect[1][1])
        rect_min = int(rotated_rect[1][1]) if rotated_rect[1][0] > rotated_rect[1][1] else int(rotated_rect[1][0])

        angle = rotated_rect[2] if abs(rotated_rect[2])<45 else (rotated_rect[2]+90)
        M = cv2.getRotationMatrix2D((int(rotated_rect[0][0]),int(rotated_rect[0][1])),angle,1)
        rotated_img[i] = cv2.warpAffine(image, M, (width,height))
        rotated_img[i] = cv2.getRectSubPix(rotated_img[i], (rect_max, rect_min),
                                        (int(rotated_rect[0][0]), int(rotated_rect[0][1])))
    return rotated_img

def seg_map_iou(image, mask, xml_path=None, png_path=None):
    '''
    单张图像分割效果评估，标注IOU
    :param image: 图像
    :param mask: 预测的mask
    :param xml_path: xml标注路径
    :param png_path: png标注路径
    :return:
    '''
    img_h = image.shape[0]
    img_w = image.shape[1]
    mask_h = mask.shape[0]
    mask_w = mask.shape[1]
    if not (img_h==mask_h and img_w==mask_w):
        mask = cv2.resize(mask, (img_w,img_h),interpolation=cv2.INTER_NEAREST)
    _, seg_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    truth_mask = []
    if xml_path:
        tree = ET.parse(xml_path)
        objs = tree.findall('object')
        truth_mask = np.zeros([img_h, img_w], dtype=np.uint8)
        for obj in objs:
            points_x = obj.find('polygen').find('points_x').text.split(',')
            points_y = obj.find('polygen').find('points_y').text.split(',')
            if points_x[-1] == '':
                point_num = len(points_x) - 1
            else:
                point_num = len(points_x)
            ps = np.zeros([point_num, 2], dtype=np.int32)
            for i in range(point_num):
                if points_x[i] == '':
                    continue
                ps[i] = (int(float(points_x[i])), int(float(points_y[i])))
            cv2.polylines(truth_mask, [ps], 1, 0, 1)  # img:图像,顶点集，是否闭合，颜色，线宽度
            cv2.fillPoly(truth_mask, [ps], 255)
    elif png_path:
        truth_mask = cv2.imdecode(np.fromfile(png_path,dtype=np.uint8),-1)
    else:
        print('没有标注')
    Union_mask = cv2.bitwise_or(truth_mask, seg_mask)
    Union_contours, hierarchy = cv2.findContours(Union_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    Union_area = 0
    if len(Union_contours) > 0:
        for i in range(len(Union_contours)):
            Union_area += cv2.contourArea(Union_contours[i])

    Intersection_mask = cv2.bitwise_and(truth_mask, seg_mask)
    Intersection_contours, hierarchy = cv2.findContours(Intersection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    Intersection_area = 0
    if len(Intersection_contours) > 0:
        for i in range(len(Intersection_contours)):
            Intersection_area += cv2.contourArea(Intersection_contours[i])

    return float(Intersection_area)/Union_area