# 一个通用的tensorflow检测模型测试框架
# 兼容模型：faster-rcnn, yolo,ssd,centernet
#   功能：在原图上绘制检测结果
#         根据检测结果裁出检测框，并按类别保存
#         计算召回率，mAP
import os
import tensorflow as tf
import cv2
import numpy as np
import xml.etree.ElementTree as ET

def create_graph(pb_path):
    '''
    创建pb模型
    :param pb_path:pb路径
    :return:
    '''
    with tf.gfile.FastGFile(pb_path, 'rb') as f:
        print('加载模型：'+pb_path)
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def My_Non_Maximum_Suppression(boxes, scores, classes, width, height):
##  输入：框、对应的分数、对应的类别
    num_boxes = boxes.shape[0]  #   框数量
    delete_box = [] #   需要删除的框
    iou_thresh = 0.4 # iou大于该值删除
    score_thresh = 0.3

    box_num, _ = np.shape(boxes)
    # 先清除分数较低的框
    deleteIndex = []
    for i in range(box_num):
        n = 1
        if scores[i] < score_thresh:
            deleteIndex.append(i)
    boxes = np.delete(boxes, deleteIndex, 0)
    scores = np.delete(scores, deleteIndex, 0)
    classes = np.delete(classes, deleteIndex, 0)
    box_num, _ = np.shape(boxes)
    boxes[:, 0] *= height
    boxes[:, 2] *= height
    boxes[:, 1] *= width
    boxes[:, 3] *= width

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

def pic_DrawBox(folder_path, crop_path=None,input_node=None,output_node_list=None):
    rootdir = os.listdir(folder_path)
    for e in rootdir:
        subdir = os.path.join(folder_path, e)
        if os.path.isfile(subdir):  # 如果是文件
            if os.path.splitext(subdir)[1] in {'.jpg', '.bmp', '.png'}:
                tf.logging.set_verbosity(tf.logging.INFO)

                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                with tf.Session(config=config) as sess:
                    image_tensor = sess.graph.get_tensor_by_name(input_node)
                    detection_boxes = sess.graph.get_tensor_by_name(output_node_list['box'])
                    detection_scores = sess.graph.get_tensor_by_name(output_node_list['score'])
                    detection_classes = sess.graph.get_tensor_by_name(output_node_list['class'])
                    image = cv2.imdecode(np.fromfile(subdir, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                    (height, width, _) =image.shape
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image_np_expanded = np.expand_dims(image, axis=0)

                    (boxes, scores, classes) = sess.run(
                        [detection_boxes, detection_scores, detection_classes],
                        feed_dict={image_tensor: image_np_expanded})
                    if boxes.ndim==3:
                        boxes = np.squeeze(boxes)
                        scores = np.squeeze(scores)
                        classes = np.squeeze(classes)

                    (boxes, scores, classes) = My_Non_Maximum_Suppression(
                        boxes,
                        scores,
                        classes.astype(np.int32),
                        width,
                        height
                    )
                    def drawBox(subdir, boxes,classes):
                        image = cv2.imdecode(np.fromfile(subdir, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                        (height, width, _) = image.shape
                        boxes[:, 0] *= height
                        boxes[:, 2] *= height
                        boxes[:, 1] *= width
                        boxes[:, 3] *= width
                        for i,box in enumerate(boxes):
                            cv2.rectangle(image, (int(box[1]),int(box[0])),(int(box[3]),int(box[2])),(0,255,0),2)
                            cv2.putText(image, str(classes[i]), (int(box[1]),int(box[0])), 1, 2, (0,0,255), 1)
                            cv2.imshow('1',image)
                        cv2.imencode('.jpg', image)[1].tofile(subdir)
                    drawBox(subdir, boxes,classes)

        elif os.path.isdir(subdir):  #   如果是路径
            pic_DrawBox(subdir, crop_path,input_node,output_node_list)

def Detect_DrawBox(pb_path, folder_path, crop_path=None,input_node=None,output_node_list=None):
    if input_node==None or output_node_list==None:
        print('请输入正确的输入输出节点\n')
    else:
        create_graph(pb_path)
        pic_DrawBox(folder_path, crop_path,input_node,output_node_list)

def pic_cropObject(folder_path, crop_path=None,input_node=None,output_node_list=None):
    rootdir = os.listdir(folder_path)
    for e in rootdir:
        subdir = os.path.join(folder_path, e)
        if os.path.isfile(subdir):  # 如果是文件
            if os.path.splitext(subdir)[1] in {'.jpg', '.bmp', '.png'}:
                tf.logging.set_verbosity(tf.logging.INFO)

                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                with tf.Session(config=config) as sess:
                    image_tensor = sess.graph.get_tensor_by_name(input_node)
                    detection_boxes = sess.graph.get_tensor_by_name(output_node_list['box'])
                    detection_scores = sess.graph.get_tensor_by_name(output_node_list['score'])
                    detection_classes = sess.graph.get_tensor_by_name(output_node_list['class'])
                    image = cv2.imdecode(np.fromfile(subdir, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image_np_expanded = np.expand_dims(image, axis=0)

                    (boxes, scores, classes) = sess.run(
                        [detection_boxes, detection_scores, detection_classes],
                        feed_dict={image_tensor: image_np_expanded})
                    if boxes.ndim==3:
                        boxes = np.squeeze(boxes)
                        scores = np.squeeze(scores)
                        classes = np.squeeze(classes)
                    def cropObject(subdir, crop_path,boxes,classes):
                        image = cv2.imdecode(np.fromfile(subdir, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                        (height, width, _) = image.shape
                        boxes[:, 0] *= height
                        boxes[:, 2] *= height
                        boxes[:, 1] *= width
                        boxes[:, 3] *= width
                        for i,box in enumerate(boxes):
                            path = os.path.join(crop_path, str(classes[i]))
                            if not os.path.exists(path):
                                os.mkdir(path)
                            crop_image = image[int(box[0]):int(box[2]),int(box[1]):int(box[3])]
                            # cv2.rectangle(image, (int(box[1]),int(box[0])),(int(box[3]),int(box[2])),(0,255,0),2)
                            # cv2.putText(image, str(classes[i]), (int(box[1]),int(box[0])), 1, 2, (0,0,255), 1)
                            # cv2.imshow('1',image)
                            try:
                                cv2.imencode('.jpg', crop_image)[1].tofile(os.path.join(path, os.path.basename(subdir)[:-4]+'_crop_'+str(i)+'.jpg'))
                            except:
                                print('find empty box!')
                    cropObject(subdir, crop_path,boxes,classes)

        elif os.path.isdir(subdir):  #   如果是路径
            pic_DrawBox(subdir, crop_path,input_node,output_node_list)

def Detect_cropObject(pb_path, folder_path, crop_path,input_node=None,output_node_list=None):
    if input_node==None or output_node_list==None:
        print('请输入正确的输入输出节点\n')
    else:
        create_graph(pb_path)
        pic_cropObject(folder_path, crop_path,input_node,output_node_list)

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
def pic_calculate_mAP(folder_path, thresh,input_node=None,output_node_list=None):
    rootdir = os.listdir(folder_path)
    for e in rootdir:
        subdir = os.path.join(folder_path, e)
        if os.path.isfile(subdir):  # 如果是文件
            if os.path.splitext(subdir)[1] in {'.jpg', '.bmp', '.png'}:
                tf.logging.set_verbosity(tf.logging.INFO)
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                with tf.Session(config=config) as sess:
                    image_tensor = sess.graph.get_tensor_by_name(input_node)
                    detection_boxes = sess.graph.get_tensor_by_name(output_node_list['box'])
                    detection_scores = sess.graph.get_tensor_by_name(output_node_list['score'])
                    detection_classes = sess.graph.get_tensor_by_name(output_node_list['class'])
                    image = cv2.imdecode(np.fromfile(subdir, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                    xml_path = subdir[:-4]+'.xml'
                    if os.path.exists(xml_path):
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        # image = image/255.
                        image_np_expanded = np.expand_dims(image, axis=0)
                        (boxes, scores, classes) = sess.run(
                            [detection_boxes, detection_scores, detection_classes],
                            feed_dict={image_tensor: image_np_expanded})
                        if boxes.ndim==3:
                            boxes = np.squeeze(boxes)
                            scores = np.squeeze(scores)
                            classes = np.squeeze(classes)
                        def calculate_mAP(subdir, thresh,boxes,classes,scores,xml_path):
                            global d  # 检测数
                            global a  # TP数
                            global b  # 目标数
                            global x

                            image = cv2.imdecode(np.fromfile(subdir, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                            (height, width, _) = image.shape
                            min_score_thresh = 0.3
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

                            d += box_num
                            tree = ET.parse(xml_path)
                            root = tree.getroot()
                            objs = root.findall('object')

                            b += len(objs)
                            gts = []
                            for obj in objs:
                                gt_ymin = int(obj.find('bndbox').find('ymin').text)
                                gt_xmin = int(obj.find('bndbox').find('xmin').text)
                                gt_ymax = int(obj.find('bndbox').find('ymax').text)
                                gt_xmax = int(obj.find('bndbox').find('xmax').text)

                                gt = [gt_ymin, gt_xmin, gt_ymax, gt_xmax]
                                gts.append(gt)

                            gts_copy = gts
                            TPs, FPs, FNs = get_TP_FP_FN(boxes, gts_copy, thresh)
                            a += len(TPs)

                        calculate_mAP(subdir, thresh,boxes,classes,scores,xml_path)

        elif os.path.isdir(subdir):  #   如果是路径
            pic_calculate_mAP(subdir, thresh,input_node,output_node_list)

def Detect_calculate_mAP(pb_path, folder_path,thresh=0.3,input_node=None,output_node_list=None):
    global d  # 检测数
    global a  # TP数
    global b  # 目标数
    global x
    a = 0
    d = 0
    b = 0
    x = 0
    if input_node==None or output_node_list==None:
        print('请输入正确的输入输出节点\n')
    else:
        create_graph(pb_path)
        pic_calculate_mAP(folder_path,thresh,input_node,output_node_list)
        recall = float(a) / b
        precision = float(a) / d
        print('召回率为:', recall, '.准确率为:', precision)