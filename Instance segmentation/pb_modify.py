# 实例分割模型节点输出结果需要经过很多tf操作才能转换成可用的实力分割mask，
# 现将这些操作写进pb

from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
import tensorflow as tf

pb_path = r'E:\玉米\pb_925/frozen_inference_graph.pb'

#加载模型
def create_graph():
    with tf.gfile.FastGFile(pb_path, 'rb') as f:
        print(pb_path)
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def main():
    create_graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        boxes = sess.graph.get_tensor_by_name('detection_boxes:0')
        num_detections = sess.graph.get_tensor_by_name('num_detections:0')
        image = sess.graph.get_tensor_by_name('image_tensor:0')
        masks = sess.graph.get_tensor_by_name('detection_masks:0')

        m_image = tf.squeeze(image, axis=0)
        Image_Size = tf.shape(m_image)[:2]
        detection_masks = tf.squeeze(masks, [0])
        detection_boxes = tf.squeeze(boxes, [0])

        real_num_detection = tf.cast(num_detections[0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, Image_Size[0], Image_Size[1])
        detection_masks = tf.cast(tf.greater(detection_masks, 0.1), tf.uint8)
        m_masks = tf.expand_dims(detection_masks, 0, name = 'Instance_Masks')

        # output_graph_def = tf.graph_util.convert_variables_to_constants(
        #     sess, sess.graph_def, output_node_names=['Instance_Masks'])
        tf.train.write_graph(sess.graph_def, '',
                             r'E:\玉米/frozen_inception_resnet_v2_ins_graph.pb', as_text=False)
if __name__ == '__main__':
  main()

