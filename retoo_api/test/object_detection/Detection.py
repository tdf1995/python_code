# 检测测试类
import tensorflow as tf
import cv2
import numpy as np


class Detection():
    def __init__(self,model_path,output_node=[],input_node='',preFunc=None, nms_thresh=None, score_thresh=None,analysisFunc=None):
        self.sess = self.model_Init(model_path)
        self.output_node = output_node
        self.input_node = input_node
        self.preFunc = preFunc
        self.analysisFunc = analysisFunc
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        if self.preFunc == None:
            print('采用默认预处理操作：RGB通道+拓展维度，如需特殊预处理请在实例化时传入！')
        if self.analysisFunc == None:
            print('采用默认的结果处理操作，boxes,scores,labels降维，并筛选、nms,如需模型特殊请在实例化时传入！')
    def model_Init(self, model_path):
        tf.logging.set_verbosity(tf.logging.ERROR)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Graph().as_default() as graph:
            with tf.gfile.FastGFile(model_path, 'rb') as f:
                print(model_path)
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(graph_def, name='')
                sess = tf.Session(graph=graph, config=config)
                return sess
    def preprocess(self, image):
        if self.preFunc == None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.expand_dims(image, axis=0)
        else:
            image = self.preFunc(image)
        return image

    def inference_detection_model(self,image):
        # Actual detection.
        self.height = image.shape[0]
        self.width = image.shape[1]
        image = self.preprocess(image)
        self.result = self.sess.run(
            self.output_node,
            feed_dict={self.input_node: image})

    # 删除分数过低的检测框
    def scores_filter(self,boxes, scores, classes, score_thresh=0.5):
        delete_box = []
        for i, score in enumerate(scores):
            if score < score_thresh:
                delete_box = np.append(delete_box, i)
        scores = np.delete(scores, delete_box)
        classes = np.delete(classes, delete_box)
        boxes = np.delete(boxes, delete_box, axis=0)

        return boxes, scores, classes

    def nms(self,boxes, scores, classes, width, height, nms_thresh=0.3):
        ##  输入：框、对应的分数、对应的类别
        num_boxes = boxes.shape[0]  # 框数量
        delete_box = []  # 保存需要删除的框的序号
        iou_thresh = nms_thresh  # nms删除阈值指标，iou大于该值删除
        for i in range(num_boxes - 1):
            for j in range(i + 1, num_boxes):
                xx1 = np.maximum(boxes[i][0] * height, boxes[j][0] * height)  # 重叠区域(如果存在)的x的最小值
                yy1 = np.maximum(boxes[i][1] * width, boxes[j][1] * width)  # 重叠区域(如果存在)的y的最小值
                xx2 = np.minimum(boxes[i][2] * height, boxes[j][2] * height)  # 重叠区域(如果存在)的x的最大值
                yy2 = np.minimum(boxes[i][3] * width, boxes[j][3] * width)  # 重叠区域(如果存在)的y的最大值

                areas1 = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) * width * height
                areas2 = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1]) * width * height

                w = np.maximum(0.0, xx2 - xx1 + 1)  # 重叠区域宽
                h = np.maximum(0.0, yy2 - yy1 + 1)  # 重叠区域高
                inter = w * h  # 重叠区域的面积
                IOU = inter / (areas1 + areas2 - inter)  # 计算IOU
                if not classes[i] == classes[j]:
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

    def results_analysis(self):
        if self.analysisFunc==None:

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

            if self.score_thresh:
                boxes, scores, classes = self.scores_filter(boxes, scores, classes, score_thresh=self.score_thresh)
            if self.nms_thresh:
                boxes, scores, classes = self.nms(boxes, scores, classes, self.width, self.height, nms_thresh=self.nms_thresh)

            return boxes, scores, classes
        else:
            self.analysisFunc()


