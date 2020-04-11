import numpy as np
import cv2
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

def od_boxes_recall(boxes,xml_path,height,width,recall_thresh=0.3):
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

def draw_boxes_in_pic(img,boxes,scores,classes,draw_scores=True,draw_classes=True):
    '''
    在原图上画框
    :param img:原图
    :param boxes: 框
    :param scores: 分数
    :param classes: 类别
    :param draw_scores: 是否画分数
    :param draw_classes: 是否画类别
    :return: 画好的图
    '''
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    height = img.shape[0]
    width = img.shape[1]
    if len(boxes)==0:
        return img
    else:
        if boxes.max()>1:
            for i, box in enumerate(boxes):
                (ymin, xmin, ymax, xmax) = (
                    int(box[0]), int(box[1] ), int(box[2]), int(box[3]))
                if ymin < 0:
                    ymin = 0
                if xmin < 0:
                    xmin = 0
                if ymax > height:
                    ymax = height
                if xmax > width:
                    xmax = width
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), colors[int(classes[i])], 2)
                if draw_classes and draw_scores:
                    cv2.putText(img, str(int(classes[i])) + ':' + str(scores[i]), (xmin - 5, ymin - 5), cv2.FONT_HERSHEY_COMPLEX,
                                0.5, colors[int(classes[i])], 1)
                if draw_classes and not draw_scores:
                    cv2.putText(img, str(int(classes[i])), (xmin - 5, ymin - 5), cv2.FONT_HERSHEY_COMPLEX,
                                0.5, colors[int(classes[i])], 1)
        else:
            for i,box in enumerate(boxes):
                (ymin, xmin, ymax, xmax) = (
                int(box[0] * height), int(box[1] * width), int(box[2] * height), int(box[3] * width))
                if ymin<0:
                    ymin = 0
                if xmin<0:
                    xmin = 0
                if ymax > height:
                    ymax =height
                if xmax > width:
                    xmax = width
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), colors[int(classes[i])], 2)
                if draw_classes and draw_scores:
                    cv2.putText(img, str(int(classes[i]))+':'+str(scores[i]), (xmin - 5, ymin - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5,colors[int(classes[i])], 1)
                if draw_classes and not draw_scores:
                    cv2.putText(img, str(int(classes[i])), (xmin - 5, ymin - 5), cv2.FONT_HERSHEY_COMPLEX,
                                0.5, colors[int(classes[i])], 1)

        return img

def crop_imgs_in_pic(image, boxes):
    '''
    在原图上根据框隔小图
    :param image: 原图
    :param boxes: 框
    :return: 小图
    '''
    height = image.shape[0]
    width = image.shape[1]
    crop_imgs = []
    if len(boxes)==0:
        return crop_imgs
    if boxes.max() > 1:
        for i, box in enumerate(boxes):
            (ymin, xmin, ymax, xmax) = (
                int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            if ymin < 0:
                ymin = 0
            if xmin < 0:
                xmin = 0
            if ymax > height:
                ymax = height
            if xmax > width:
                xmax = width

            crop_imgs.append(image[ymin:ymax,xmin:xmax])
    else:
        for i, box in enumerate(boxes):
            (ymin, xmin, ymax, xmax) = (
                int(box[0] * height), int(box[1] * width), int(box[2] * height), int(box[3] * width))
            if ymin < 0:
                ymin = 0
            if xmin < 0:
                xmin = 0
            if ymax > height:
                ymax = height
            if xmax > width:
                xmax = width

            crop_imgs.append(image[ymin:ymax, xmin:xmax])
    return crop_imgs