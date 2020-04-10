# 批量目标检测，统计评估指标：准确率、召回率。此为单类别，多类别有具体项目再写
import os
import glob
import cv2
import numpy as np
from detect_faster_rcnn import detector_faster_rcnn
from detect_yolo import detector_yolo
from detection_assist import *
import xml.etree.ElementTree as ET

def iou(dt, gt):# dt:检测框，gt:目标框
    S_rec_gt = (gt[2] - gt[0]) * (gt[3] - gt[1]) # 标注框的面积
    S_rec_dt = (dt[2] - dt[0]) * (dt[3] - dt[1]) # 预测框的面积

    sum_area = S_rec_dt + S_rec_gt

    left_line = max(dt[1], gt[1])
    right_line = min(dt[3], gt[3])
    top_line = max(dt[0], gt[0])
    bottom_line = min(dt[2], gt[2])

    if left_line >= right_line or top_line >= bottom_line:
        intersect = 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
    return float(intersect) / float(sum_area - intersect)

def get_TP_FP_FN(dts, gts_copy, iou_thre):
    # faster rcnn检测的框排序是按分数排序的
    TPs = []
    FPs = []
    for dt in dts:
        if len(gts_copy):
            max_iou = max([iou(dt, gt) for gt in gts_copy])
            max_iou_gt = np.argmax([iou(dt, gt) for gt in gts_copy]) # 对于每个检测框找到最大的iou和对应的gt框
            if max_iou >= iou_thre:
                TPs.append(dt)  # 当做一个正确检测
                gts_copy = np.delete(gts_copy, max_iou_gt, 0) # gt框不会被匹配两次
    tp_num = len(TPs)
    for dt in dts:
        for i in range(tp_num):
            if (dt == TPs[i]).all() :# 剩下的无论是多余的检测框还是检测错的检测框，标记为FP
                break
            elif i+1 == tp_num:
                FPs.append(dt)

    # FPs = [dt for dt in dts if not dt in TPs]
    FNs = gts_copy   # 没有被匹配的标记为FN

    return TPs, FPs, FNs

def od_boxes_recall(boxes,xml_path,recall_thresh=0.3):
    '''
    :param boxes: 检测框
    :param xml_path: 标注文件位置
    :param recall_thresh: iou阈值，判断是否是正确检测
    :return: 真实框，检测框，正确检测框
    '''

    # 需要转化为绝对坐标格式
    if len(boxes) and not boxes.max()>1:
        boxes[:, 0] *= height
        boxes[:, 2] *= height
        boxes[:, 1] *= width
        boxes[:, 3] *= width

    tree = ET.parse(xml_path)
    root = tree.getroot()
    objs = root.findall('object')
    gt_boxes = []
    for obj in objs:
        gt_ymin = int(obj.find('bndbox').find('ymin').text)
        gt_xmin = int(obj.find('bndbox').find('xmin').text)
        gt_ymax = int(obj.find('bndbox').find('ymax').text)
        gt_xmax = int(obj.find('bndbox').find('xmax').text)

        gt = [gt_ymin, gt_xmin, gt_ymax, gt_xmax]
        gt_boxes.append(gt)
    gt_boxes_copy = gt_boxes
    tp_boxes, fp_boxes, fn_boxes = get_TP_FP_FN(boxes, gt_boxes_copy, recall_thresh)

    return gt_boxes,boxes,tp_boxes

if __name__ == '__main__':
    model_path = r'F:\models-master\cell\pb_726\frozen_inference_graph.pb'
    det = detector_faster_rcnn(model_path)
    img_root_path = r'F:\models-master\test_set\JPEGImages'
    xml_root_path = r'F:\models-master\test_set\Annotations'
    recall_thresh = 0.3

    img_paths = glob.glob(img_root_path + '/*.jpg')
    xml_paths = glob.glob(xml_root_path + '/*.xml')


    truth_total = 0
    detect_total = 0
    truth_positive_total = 0

    for img_path in img_paths:
        match_xml_path = os.path.join(xml_root_path,os.path.basename(img_path)[:-4]+'.xml')
        if not match_xml_path in xml_paths:
            print("can't match {} in {},skipped already".format(os.path.basename(match_xml_path),xml_root_path))
            continue
        else:
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
            height = img.shape[0]
            width = img.shape[1]
            det.inference_detection_model(img)
            boxes, scores, classes = det.result_analysis()
            boxes, scores, classes = scores_filter(boxes, scores, classes, score_thresh=0.4)
            boxes, scores, classes = nms(boxes, scores, classes, width, height, 0.4)

            gt_boxes, det_boxes, tp_boxes = od_boxes_recall(boxes, match_xml_path,recall_thresh)
            truth_total += len(gt_boxes)
            detect_total += len(det_boxes)
            truth_positive_total += len(tp_boxes)
            print(truth_total)
            print(detect_total)
            print(truth_positive_total)

    recall = float(truth_positive_total) / truth_total
    precision = float(truth_positive_total) / detect_total
    print('iou为{}召回率为:'.format(recall_thresh), recall, '.准确率为:', precision)
