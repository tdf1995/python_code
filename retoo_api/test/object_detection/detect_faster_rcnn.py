import numpy as np
import cv2
from Detection import Detection
from detection_assist import *

class detector_faster_rcnn(Detection):
    def __init__(self,model_path,output_node=['detection_boxes:0','detection_scores:0','detection_classes:0'],
                 input_node='image_tensor:0',preFunc=None):
        super(detector_faster_rcnn, self).__init__(model_path)
        self.sess = self.model_Init(model_path)
        self.output_node = output_node
        self.input_node = input_node
        self.preFunc = preFunc

    def result_analysis(self):
        boxes_ndim = self.result[0].ndim
        scores_ndim = self.result[1].ndim
        classes_ndim = self.result[2].ndim
        if boxes_ndim > 2:
            boxes = np.squeeze(self.result[0])
        else:
            boxes = self.result[0]
        if scores_ndim > 1:
            scores = np.squeeze(self.result[1])
        else:
            scores = self.result[1]
        if classes_ndim > 1:
            classes = np.squeeze(self.result[2])
        else:
            classes = self.result[2]
        return boxes, scores, classes


if __name__ == '__main__':

    def preprocess(image):
        print(1)
        return image

    pb_path = r'E:\ocr\检测\pb_0227\frozen_inference_graph.pb'
    img_path = r'E:\ocr\检测\test_pic\2019_04_19_22_41_12_662_roi_crop.jpg'
    det = detector_faster_rcnn(pb_path)
    image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    height = image.shape[0]
    width = image.shape[1]
    det.inference_detection_model(image)
    boxes,scores,classes = det.result_analysis()
    boxes, scores, classes = scores_filter(boxes, scores, classes,score_thresh=0.95)
    boxes, scores, classes = nms(boxes, scores, classes, width, height, 0.3)
    image = draw_boxes_in_pic(image, boxes, scores, classes)
    cv2.imshow('1', image)
    cv2.waitKey(0)
    print(1)