# 批量语义分割，统计评估指标：IOU(交并比)、PA(像素精度),此为单类别分割，多类别有实际项目需求再写
import os
import glob
import cv2
import numpy as np
from Segmentation import Segmentation
from segmentation_assist import *


def batch_seg_test_IOU(seg, img_root_path, xml_root_path=None,png_root_path=None):
    '''
    批量测试分割模型的准确率，标准为IOU
    :param seg: 分割模型
    :param img_root_path:原图根目录
    :param xml_root_path: 标注根目录
    :param png_root_path: mask标注根目录
    :return:
    '''

    img_paths = glob.glob(img_root_path + '/*.jpg')
    if xml_root_path:
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
                IOU_total += seg_map_iou(img, mask, xml_path=match_xml_path)
                num_total += 1

        print("{}张图总体IOU:".format(num_total), IOU_total / num_total)

    elif png_root_path:
        png_paths = glob.glob(png_root_path + '/*.png')
        IOU_total = 0
        num_total = 0
        for img_path in img_paths:
            match_png_path = os.path.join(xml_root_path, os.path.basename(img_path)[:-4] + '.png')
            if not match_png_path in png_paths:
                print("can't match {} in {},skipped already".format(os.path.basename(match_png_path), png_root_path))
                continue
            else:
                img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
                height = img.shape[0]
                width = img.shape[1]
                seg.inference_detection_model(img)
                mask = seg.results_analysis()
                IOU_total += seg_map_iou(img, mask, png_path=match_png_path)
                num_total += 1

        print("{}张图总体IOU:".format(num_total), IOU_total / num_total)
    else:
        print('没有给定标注路径')

def batch_mask_save(seg, img_root_path, target_root_path=None, Visualization=False):
    '''
    批量生成分割mask到制定路径
    :param seg: 分割模型
    :param img_root_path:图像根目录
    :param target_root_path:结果保存路径
    :return:
    '''
    rootdir = os.listdir(img_root_path)
    for e in rootdir:
        subdir = os.path.join(img_root_path, e)  # 子文件及子文件夹路径
        if os.path.isfile(subdir):  # 如果是文件
            if os.path.splitext(subdir)[1] in {'.jpg', '.bmp', '.png'}:
                image = cv2.imdecode(np.fromfile(subdir, dtype=np.uint8), -1)
                print('\r', subdir, end='', flush=True)
                seg.inference_detection_model(image)
                mask = seg.results_analysis()
                if Visualization:
                    mask = mask*255
                if target_root_path:
                    if not os.path.exists(target_root_path):
                        os.makedirs(target_root_path)
                    target_file_path = os.path.join(target_root_path, e)
                    cv2.imencode('.png', mask)[1].tofile(target_file_path[:-4] + '.png')
                else:
                    cv2.imencode('.png', mask)[1].tofile(subdir[:-4] + '.png')
        elif os.path.isdir(subdir):  # 如果是路径
            batch_mask_save(seg, subdir, target_root_path)

if __name__ == '__main__':
    pb_path = r'E:\菜品\分割\train_data\pb_1212\frozen_inference_graph_caiping_1212.pb'
    img_root_path = r'E:\菜品\分割\训练标注数据\第一批'
    xml_root_path = r'E:\菜品\分割\训练标注数据\第一批'
    def preprocess(image):
        image = cv2.resize(image, (600, 400), interpolation=cv2.INTER_NEAREST)
        image = np.expand_dims(image, axis=0)
        return image

    seg = Segmentation(pb_path, preFunc=preprocess)

    batch_mask_save(seg, img_root_path, './test/',Visualization=True)
    # batch_seg_test_IOU(seg,img_root_path,xml_root_path=xml_root_path)


