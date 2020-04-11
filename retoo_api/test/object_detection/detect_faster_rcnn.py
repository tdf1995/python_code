import numpy as np
import cv2
from Detection import Detection
from detection_assist import *

class detector_faster_rcnn(Detection):
    def __init__(self,model_path,output_node=['detection_boxes:0','detection_scores:0','detection_classes:0'],
                 input_node='image_tensor:0',preFunc=None, nms_thresh=None, score_thresh=None):
        super(detector_faster_rcnn, self).__init__(model_path)

        self.output_node = output_node
        self.input_node = input_node
        self.preFunc = preFunc
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh


# if __name__ == '__main__':
#
#     def preprocess(image):
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = cv2.resize(image,(1200,128),cv2.INTER_NEAREST)
#         image = np.expand_dims(image, 0)
#         return image
#
#     pb_path = r'E:\ocr\检测\pb_0227\frozen_inference_graph.pb'
#     img_path = r'E:\ocr\检测\test_pic\2019_04_19_22_41_12_662_roi_crop.jpg'
#     det = detector_faster_rcnn(pb_path,preFunc=preprocess,nms_thresh=0.3, score_thresh=0.95)
#     image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
#     height = image.shape[0]
#     width = image.shape[1]
#     det.inference_detection_model(image)
#     boxes,scores,classes = det.results_analysis()
#
#     image = draw_boxes_in_pic(image, boxes, scores, classes)
#     cv2.imshow('2',image)
#     crops = crop_imgs_in_pic(image, boxes)
#     for crop in crops:
#         cv2.imshow('1', crop)
#         cv2.waitKey(0)
#     print(1)