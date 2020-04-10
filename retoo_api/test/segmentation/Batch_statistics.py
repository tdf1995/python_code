# 批量语义分割，统计评估指标：IOU(交并比)、PA(像素精度),此为单类别分割，多类别有实际项目需求再写
import os
import glob
import cv2
import numpy as np
from Segmentation import Segmentation
from segmentation_assist import *
import xml.etree.ElementTree as ET

def seg_map_iou(image, mask, xml_path):
    img_h = image.shape[0]
    img_w = image.shape[1]
    mask_h = mask.shape[0]
    mask_w = mask.shape[1]
    if not (img_h==mask_h and img_w==mask_w):
        mask = cv2.resize(mask, (img_w,img_h),interpolation=cv2.INTER_NEAREST)
    _, seg_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

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
    # Union_mask = cv2.resize(Union_mask, (600, 400))
    # Intersection_mask = cv2.resize(Intersection_mask, (600, 400))
    # seg_mask = cv2.resize(seg_mask,(600,400))
    # truth_mask = cv2.resize(truth_mask, (600, 400))
    # cv2.imshow('0', Union_mask)
    # cv2.imshow('1', Intersection_mask)
    # cv2.imshow('2',seg_mask)
    # cv2.imshow('3',truth_mask)
    # cv2.waitKey(0)
    # print(float(Intersection_area)/Union_area)
    return float(Intersection_area)/Union_area


if __name__ == '__main__':
    pb_path = r'E:\菜品\分割\train_data\pb_1212\frozen_inference_graph_caiping_1212.pb'
    img_root_path = r'E:\菜品\分割\训练标注数据\第一批'
    xml_root_path = r'E:\菜品\分割\训练标注数据\第一批'
    def preprocess(image):
        image = cv2.resize(image, (600, 400), interpolation=cv2.INTER_NEAREST)
        image = np.expand_dims(image, axis=0)
        return image
    seg = Segmentation(pb_path, preFunc=preprocess)

    img_paths = glob.glob(img_root_path + '/*.jpg')
    xml_paths = glob.glob(xml_root_path + '/*.xml')

    IOU_total = 0
    num_total = 0
    for img_path in img_paths:
        match_xml_path = os.path.join(xml_root_path, os.path.basename(img_path)[:-4] + '.xml')
        if not match_xml_path in xml_paths:
            print("can't match {} in {},skipped already".format(os.path.basename(match_xml_path), xml_root_path))
            continue
        else:
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
            height = img.shape[0]
            width = img.shape[1]
            seg.inference_detection_model(img)
            mask = seg.results_analysis()
            IOU_total += seg_map_iou(img, mask, match_xml_path)
            num_total +=1

    print("{}张图总体IOU:".format(num_total),IOU_total/num_total)


