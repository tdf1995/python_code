# coding: utf-8
# 为未标注的图像生成xml文件功能
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import cv2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import glob
import time
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
from lxml.etree import Element, SubElement, tostring
import math

# pb路径
PATH_TO_CKPT = r'F:\models-master\cell\pb_523/frozen_inference_graph.pb'
# 类别文件
PATH_TO_LABELS = os.path.join('F:/models-master/models-master/research/object_detection/data', 'cell_label_map.pbtxt')
# 测试图像路径
TEST_IMAGE_PATHS = glob.glob(r'F:\models-master\cell\0529_test/*.jpg')
NUM_CLASSES = 1
# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def My_Non_Maximum_Suppression(boxes, scores, classes, width, height):
##  输入：框、对应的分数、对应的类别
    num_boxes = boxes.shape[0]  #   框数量
    delete_box = [] #   需要删除的框
    iou_thresh = 0.7 # iou大于该值删除
    for i in range(num_boxes - 1):
        for j in range(i + 1, num_boxes):
            xx1 = np.maximum(boxes[i][0]*height, boxes[j][0]*height)  #   重叠区域(如果存在)的x的最小值
            yy1 = np.maximum(boxes[i][1]*width, boxes[j][1]*width)  #   重叠区域(如果存在)的y的最小值
            xx2 = np.minimum(boxes[i][2]*height, boxes[j][2]*height)  #   重叠区域(如果存在)的x的最大值
            yy2 = np.minimum(boxes[i][3]*width, boxes[j][3]*width)  #   重叠区域(如果存在)的y的最大值

            areas1 = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])*width*height
            areas2 = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])*width*height

            w = np.maximum(0.0, xx2 - xx1+1)  # 重叠区域宽
            h = np.maximum(0.0, yy2 - yy1+1)  # 重叠区域高
            inter = w * h # 重叠区域的面积
            IOU = inter / (areas1 + areas2 - inter) # 计算IOU
            if not classes[i]==classes[j]:
                continue
            elif IOU <= iou_thresh:
                continue
            elif scores[i] > scores[j]:
                delete_box = np.append(delete_box, j)
            else:
                delete_box = np.append(delete_box, i)


    scores = np.delete(scores, delete_box)
    classes = np.delete(classes, delete_box)
    boxes = np.delete(boxes, delete_box, axis=0)
    return boxes, scores, classes

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.shape
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)
# # Detection

def Write_Detection(image, boxes, classes, scores, image_path):
    min_score_thresh = 0.4
    (height, width, _) = image.shape
    box_num, _ = np.shape(boxes)

    # 先清除分数较低的框
    deleteIndex = []
    for i in range(box_num):
        n = 1
        if scores[i] < min_score_thresh:
            deleteIndex.append(i)
    boxes = np.delete(boxes, deleteIndex, 0)
    scores = np.delete(scores, deleteIndex, 0)
    classes = np.delete(classes, deleteIndex, 0)
    box_num, _ = np.shape(boxes)
    boxes[:, 0] *= height
    boxes[:, 2] *= height
    boxes[:, 1] *= width
    boxes[:, 3] *= width

    # 开始写xml
    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'VOC'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = os.path.basename(image_path)

    node_path = SubElement(node_root, 'path')
    node_path.text = image_path

    node_source = SubElement(node_root, 'source')
    node_database = SubElement(node_source, 'database')
    node_database.text = 'tdf'

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(width)
    node_height = SubElement(node_size, 'height')
    node_height.text = str(height)
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    for j in range(len(boxes)):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = 'danhe'
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'

        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(int(round(boxes[j, 1])))
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(int(round(boxes[j, 0])))
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(int(round(boxes[j, 3])))
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(int(round(boxes[j, 2])))
    xml = tostring(node_root, pretty_print= True)
    dom = parseString(xml)

    return dom


def main():
    # Size, in inches, of the output images.1
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            #while True:  # for image_path in TEST_IMAGE_PATHS:    #changed 20170825
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            for image_path in TEST_IMAGE_PATHS:
                time_start = time.time()
                image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_name = os.path.basename(image_path)


                (height, width, _) = image.shape
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = image
                # image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # Visualization of the results of a detection.
                (boxes, scores, classes) = My_Non_Maximum_Suppression(
                    np.squeeze(boxes),
                    np.squeeze(scores),
                    np.squeeze(classes).astype(np.int32),
                    width,
                    height
                )
                time_end = time.time()
                # vis_util.visualize_boxes_and_labels_on_image_array(
                #     image_np,
                #     boxes,
                #     classes,
                #     scores,
                #     category_index,
                #     use_normalized_coordinates=True,
                #     line_thickness=3)
                time_cost = time_end - time_start
                print('Detect single picture in {:.3f}s'.format(
                    time_cost))

                dom = Write_Detection(image, boxes, classes, scores, image_path)

                xml_path = os.path.join(os.path.dirname(image_path), image_name[:-4]+'.xml')
                with open(xml_path, 'wb') as f:
                    f.write(dom.toprettyxml(indent='\t', encoding='utf-8'))
                # cv2.imwrite(r'F:\models-master\cell\0529_test_result/'+os.path.basename(image_path),image_np)

if __name__ == '__main__':
    main()
