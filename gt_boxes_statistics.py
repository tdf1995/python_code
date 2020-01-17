# coding: utf-8
# 统计细胞标准框的各项数据:scale, ratio, num
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf
import zipfile
from collections import defaultdict
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import shutil
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import glob
import time
import xml.etree.ElementTree as ET

# xml文件
PATH_TO_GT_XML = glob.glob(r'F:\models-master\cell\VOCdevkit\VOC2007\Annotations\*.xml')
# xml文件路径
GT_XML = r'F:\models-master\test_set\Annotations'

def StatisticalDetection(boxes, xml_path):
    global a  # 已标注&已检测
    global b  # 已标注&未检测
    global c  # 未标注&已检测
    global d  # 检测数
    global e  # 目标数
    global x

    min_score_thresh = 0.4
    iou_thresh = 0.3

    box_num, _ = np.shape(boxes)
    count = box_num

    d += box_num
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objs = root.findall('object')
    e += len(objs)

    # 对于每个标注框，查找有没有匹配的预测框
    for obj in objs:
        min_distance = 1000

        gt_ymin = int(obj.find('bndbox').find('ymin').text)
        gt_xmin = int(obj.find('bndbox').find('xmin').text)
        gt_ymax = int(obj.find('bndbox').find('ymax').text)
        gt_xmax = int(obj.find('bndbox').find('xmax').text)
        gt_center_x = (gt_xmin + gt_xmax)/2
        gt_center_y = (gt_ymax + gt_ymin)/2

        S_rec_gt = (gt_xmax - gt_xmin) * (gt_ymax - gt_ymin)
        if S_rec_gt < 20:
            print(xml_path)
        else:
            FindIndex = -1
            for i in range(len(boxes)):
                S_rec_pre = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
                pre_center_x = (boxes[i, 3] + boxes[i, 1])/2
                pre_center_y = (boxes[i, 2] + boxes[i, 0])/2
                distance = np.sqrt(np.square(gt_center_x - pre_center_x) + np.square(gt_center_y - pre_center_y))

                sum_area = S_rec_pre + S_rec_gt

                left_line = max(boxes[i, 1], gt_xmin)
                right_line = min(boxes[i, 3], gt_xmax)
                top_line = max(boxes[i, 0], gt_ymin)
                bottom_line = min(boxes[i, 2], gt_ymax)

                if left_line >= right_line or top_line >= bottom_line:
                    intersect = 0
                else:
                    intersect = (right_line - left_line) * (bottom_line - top_line)
                if float(intersect) / float(S_rec_gt) > iou_thresh:
                    if distance < min_distance:
                        FindIndex = i
            if not FindIndex == -1:
                boxes = np.delete(boxes, FindIndex, 0)
                a += 1

    box_num, _ = np.shape(boxes)
    c += box_num



def main():
    Cell_Num = []
    Side_Length = []
    Ratio = []
    for gt_path in PATH_TO_GT_XML:
        gt_xml_name = os.path.basename(gt_path)
        tree = ET.parse(gt_path)
        root = tree.getroot()
        objs = root.findall('object')
        Cell_Num = np.append(Cell_Num, len(objs)) # 单个图像下细胞的数目
        for obj in objs:
            x_side = int(obj.find('bndbox').find('xmax').text) - int(obj.find('bndbox').find('xmin').text)
            y_side = int(obj.find('bndbox').find('ymax').text) - int(obj.find('bndbox').find('ymin').text)
            area = x_side * y_side
            if area < 5:
                print(gt_path)
            if x_side and y_side:
                ratio = y_side / x_side
                Side_Length = np.append(Side_Length, [x_side, y_side])
                Ratio = np.append(Ratio, ratio)
            else:
                print(gt_path)
    Side_Length = np.sort(Side_Length)
    Ratio = np.sort(Ratio)
    Cell_Num = np.sort(Cell_Num)
    Maximum_side_length = np.max(Side_Length)
    Minimum_side_length = np.min(Side_Length)
    Max_ratio = np.max(Ratio)
    Min_ratio = np.min(Ratio)
    Max_num = np.max(Cell_Num)
    print('最大边长:',Maximum_side_length)
    print('最小边长:', Minimum_side_length)
    print('最大比率:', Max_ratio)
    print('最小比率:', Min_ratio)
    print('最多细胞数:', Max_num)


if __name__ == '__main__':
    main()
