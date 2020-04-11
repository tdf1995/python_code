# 分割测试类
import tensorflow as tf
import cv2
import numpy as np
from segmentation.segmentation_assist import *

class Segmentation():
    def __init__(self, model_path,output_node='SemanticPredictions:0',input_node='ImageTensor:0',preFunc=None):
        self.sess = self.model_Init(model_path)
        self.output_node = output_node
        self.input_node = input_node
        self.preFunc = preFunc
        if self.preFunc == None:
            print('采用默认预处理操作：RGB通道+除255再转(-1,1)+拓展维度，如需特殊预处理请在实例化时传入！')

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
            # print('采用默认预处理操作：RGB通道+除255再转(-1,1)+拓展维度，如需特殊预处理请在实例化时传入！')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.expand_dims(image, axis=0)
        else:
            image = self.preFunc(image)
        return image

    def inference_detection_model(self,image):
        # Actual detection.
        image = self.preprocess(image)
        self.result = self.sess.run(
            self.output_node,
            feed_dict={self.input_node: image})

    def results_analysis(self):
        seg_map = self.result[0].astype(np.uint8)
        return seg_map

if __name__ == '__main__':

    pb_path = r'E:\菜品\分割\train_data\pb_1212\frozen_inference_graph_caiping_1212.pb'
    img_path = r'E:\菜品\分割\测试记录\test\20191210\2019.12.09 17.40.48.021.jpg'
    def preprocess(image):
        image = cv2.resize(image, (600, 400), interpolation=cv2.INTER_AREA)
        image = np.expand_dims(image, axis=0)
        return image
    seg = Segmentation(pb_path,preFunc=preprocess)
    image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    height = image.shape[0]
    width = image.shape[1]
    seg.inference_detection_model(image)
    result = seg.results_analysis()

    mask_to_show = simple_show_mask(result)

    drawed_img = draw_mask_in_img(image,result)

    crop = crop_imgs(image,result)
    crop[0] = cv2.resize(crop[0],(600,400),cv2.INTER_NEAREST)
    cv2.imshow('1',crop[0])
    cv2.imshow('2', mask_to_show)
    cv2.imshow('3', drawed_img)
    cv2.waitKey(0)