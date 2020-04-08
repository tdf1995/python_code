# 检测测试类
import tensorflow as tf
import cv2
import numpy as np

class Detection():
    def __init__(self,model_path,output_node=[],input_node='',preFunc=None):
        self.sess = self.model_Init(model_path)
        self.output_node = output_node
        self.input_node = input_node
        self.preFunc = preFunc

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
            print('采用默认预处理操作：RGB通道+拓展维度，如需特殊预处理请在实例化时传入！')
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
        pass



