import numpy as np
import cv2

# 删除分数过低的检测框
def scores_filter(boxes, scores, classes, score_thresh=0.5):
    delete_box = []
    for i,score in enumerate(scores):
        if score < score_thresh:
            delete_box = np.append(delete_box, i)
    scores = np.delete(scores, delete_box)
    classes = np.delete(classes, delete_box)
    boxes = np.delete(boxes, delete_box, axis=0)

    return boxes, scores, classes

def nms(boxes, scores, classes, width, height, thresh=0.3):
##  输入：框、对应的分数、对应的类别
    num_boxes = boxes.shape[0]  #   框数量
    delete_box = [] #   保存需要删除的框的序号
    iou_thresh = thresh # nms删除阈值指标，iou大于该值删除
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

def draw_boxes_in_pic(img,boxes,scores,classes,draw_scores=True,draw_classes=True):
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    height = img.shape[0]
    width = img.shape[1]
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
                cv2.putText(img, str(classes[i]) + ':' + str(scores[i]), (xmin - 5, ymin - 5), cv2.FONT_HERSHEY_COMPLEX,
                            0.5, colors[int(classes[i])], 1)
            if draw_classes and not draw_scores:
                cv2.putText(img, str(classes[i]), (xmin - 5, ymin - 5), cv2.FONT_HERSHEY_COMPLEX,
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
                cv2.putText(img, str(classes[i])+':'+str(scores[i]), (xmin - 5, ymin - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5,colors[int(classes[i])], 1)
            if draw_classes and not draw_scores:
                cv2.putText(img, str(classes[i]), (xmin - 5, ymin - 5), cv2.FONT_HERSHEY_COMPLEX,
                            0.5, colors[int(classes[i])], 1)

    return img

