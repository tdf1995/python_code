import numpy as np
import cv2
from Detection import Detection
from detection_assist import draw_boxes_in_pic,nms,scores_filter

class detector_yolo(Detection):
    def __init__(self,model_path,output_node=['lambda_1/concat_8:0','lambda_1/concat_9:0','lambda_1/concat_10:0'],input_node='input_1:0'
                 ,preFunc=None):
        super(detector_yolo, self).__init__(model_path)
        self.sess = self.model_Init(model_path)
        self.output_node = output_node
        self.input_node = input_node
        self.preFunc = preFunc

    def results_analysis(self):
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
        print('个人预处理')
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image/255.
        image = np.expand_dims(image, axis=0)
        return image
    pb_path = r'C:\Users\tdf\Documents\Visual Studio 2015\Projects\deepsort\deepsort\public\yolo_voc2012.pb'
    img_path = r'E:\多目标跟踪\检测训练挑选\00000026.jpg'
    det = detector_yolo(pb_path,preFunc=preprocess)
    image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    height = image.shape[0]
    width = image.shape[1]
    det.inference_detection_model(image)
    boxes,scores,classes = det.results_analysis()
    boxes, scores, classes = scores_filter(boxes, scores, classes)
    boxes, scores, classes = nms(boxes,scores,classes,width,height,0.3)
    image = draw_boxes_in_pic(image, boxes, scores, classes)
    cv2.imshow('1',image)
    cv2.waitKey(0)
    print(1)