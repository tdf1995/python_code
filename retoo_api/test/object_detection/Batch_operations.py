# 批量目标检测，统计评估指标：准确率、召回率。此为单类别，多类别有具体项目再写
import os
import glob
import cv2
import numpy as np
from detect_faster_rcnn import detector_faster_rcnn
from detect_yolo import detector_yolo
from detection_assist import *


def batch_draw_boxes(od,img_root_path,target_path=None):
    '''
    批量测试图，画检测框
    :param od: 检测模型
    :param img_root_path: 图像根目录
    :param target_path: 存放根目录
    :return:
    '''

    rootdir = os.listdir(img_root_path)
    for e in rootdir:
        subdir = os.path.join(img_root_path, e)  # 子文件及子文件夹路径
        if os.path.isfile(subdir):  # 如果是文件
            if os.path.splitext(subdir)[1] in {'.jpg', '.bmp', '.png'}:
                image = cv2.imdecode(np.fromfile(subdir, dtype=np.uint8), -1)
                print('\r',subdir,end='', flush=True)
                od.inference_detection_model(image)
                boxes, scores, classes = od.results_analysis()
                image = draw_boxes_in_pic(image, boxes, scores, classes, draw_scores=True, draw_classes=True)
                if target_path:
                    if not os.path.exists(target_path):
                        os.makedirs(target_path)
                    target_file_path = os.path.join(target_path, e)
                    if os.path.splitext(subdir)[1] == '.jpg':
                        cv2.imencode('.jpg', image)[1].tofile(target_file_path[:-4]+'_od.jpg')
                    if os.path.splitext(subdir)[1] == '.bmp':
                        cv2.imencode('.bmp', image)[1].tofile(target_file_path[:-4]+'_od.bmp')
                    if os.path.splitext(subdir)[1] == '.png':
                        cv2.imencode('.png', image)[1].tofile(target_file_path[:-4]+'_od.png')
                else:
                    if os.path.splitext(subdir)[1] == '.jpg':
                        cv2.imencode('.jpg', image)[1].tofile(subdir[:-4]+'_od.jpg')
                    if os.path.splitext(subdir)[1] == '.bmp':
                        cv2.imencode('.bmp', image)[1].tofile(subdir[:-4]+'_od.bmp')
                    if os.path.splitext(subdir)[1] == '.png':
                        cv2.imencode('.png', image)[1].tofile(subdir[:-4]+'_od.png')
        elif os.path.isdir(subdir):  # 如果是路径
            batch_draw_boxes(od, subdir, target_path)

def batch_od_recall(od, img_root_path,xml_root_path):
    '''
    统计有标注的图像的模型测试召回率
    :param od: 检测模型
    :param img_root_path: 存放图像的路径
    :param xml_root_path: 存放xml的路径
    :return:
    '''
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
            od.inference_detection_model(img)
            boxes, scores, classes = od.results_analysis()
            gt_boxes, det_boxes, tp_boxes = od_boxes_recall(boxes, match_xml_path,height,width,recall_thresh=0.3)
            truth_total += len(gt_boxes)
            detect_total += len(det_boxes)
            truth_positive_total += len(tp_boxes)

    recall = float(truth_positive_total) / truth_total
    precision = float(truth_positive_total) / detect_total
    print('iou为{}召回率为:'.format(0.3), recall, '.准确率为:', precision)

def batch_od_crop_clfmove(od, img_root_path, target_path=None):
    '''
        批量测试图，crop检测框内的图片并按类别保存
        :param od: 检测模型
        :param img_root_path: 图像根目录
        :param target_path: 存放根目录
        :return:
        '''

    rootdir = os.listdir(img_root_path)
    for e in rootdir:
        subdir = os.path.join(img_root_path, e)  # 子文件及子文件夹路径
        if os.path.isfile(subdir):  # 如果是文件
            if os.path.splitext(subdir)[1] in {'.jpg', '.bmp', '.png'}:
                print('\r',subdir,e,end='',flush=True)
                image = cv2.imdecode(np.fromfile(subdir, dtype=np.uint8), -1)
                od.inference_detection_model(image)
                boxes, scores, classes = od.results_analysis()

                crop_imgs = crop_imgs_in_pic(image, boxes)
                if not len(crop_imgs):
                    continue
                for i in range(len(crop_imgs)):
                    if target_path:
                        crop_path = os.path.join(target_path, str(classes[i]))
                        if not os.path.exists(crop_path):
                            os.makedirs(crop_path)
                        crop_img_path = os.path.join(crop_path, e)
                        if os.path.splitext(subdir)[1] == '.jpg':
                            cv2.imencode('.jpg', crop_imgs[i])[1].tofile(crop_img_path[:-4] +'_'+str(i)+ '_od.jpg')
                        if os.path.splitext(subdir)[1] == '.bmp':
                            cv2.imencode('.bmp', crop_imgs[i])[1].tofile(crop_img_path[:-4] +'_'+str(i)+ '_od.bmp')
                        if os.path.splitext(subdir)[1] == '.png':
                            cv2.imencode('.png', crop_imgs[i])[1].tofile(crop_img_path[:-4] +'_'+str(i)+ '_od.png')
                    else:
                        crop_path = os.path.join(img_root_path, str(classes[i]))
                        if not os.path.exists(crop_path):
                            os.makedirs(crop_path)
                        crop_img_path = os.path.join(crop_path, e)
                        if os.path.splitext(subdir)[1] == '.jpg':
                            cv2.imencode('.jpg', crop_imgs[i])[1].tofile(crop_img_path[:-4] +'_'+str(i)+ '_od.jpg')
                        if os.path.splitext(subdir)[1] == '.bmp':
                            cv2.imencode('.bmp', crop_imgs[i])[1].tofile(crop_img_path[:-4] +'_'+str(i)+ '_od.bmp')
                        if os.path.splitext(subdir)[1] == '.png':
                            cv2.imencode('.png', crop_imgs[i])[1].tofile(crop_img_path[:-4] +'_'+str(i)+ '_od.png')
        elif os.path.isdir(subdir):  # 如果是路径
            batch_od_crop_clfmove(od, subdir, target_path)

if __name__ == '__main__':
    model_path = r'F:\models-master\cell\pb_726/frozen_inference_graph.pb'
    det = detector_faster_rcnn(model_path,nms_thresh=0.4, score_thresh=0.4)
    img_root_path = r'F:\models-master\test_set\JPEGImages'
    xml_root_path = r'F:\models-master\test_set\Annotations'



    batch_draw_boxes(det, img_root_path,'./test/')
    # batch_od_recall(det,img_root_path,xml_root_path)
    # batch_od_crop_clfmove(det, img_root_path,r'D:\python code\retoo_api\test\object_detection\test')
