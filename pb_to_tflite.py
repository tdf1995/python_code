# tf 1.14版本以上
import tensorflow as tf

in_path = "/home/fengshijia/mars(1).pb"
out_path = "/home/fengshijia/mars(1).tflite"


input_tensor_name = ["images"]
input_tensor_shape = {"images":[1,128,64,3]}

#classes_tensor_name = ['lambda_1/box', 'lambda_1/scores', 'lambda_1/class']
classes_tensor_name = ["features"]

converter = tf.lite.TFLiteConverter.from_frozen_graph(in_path,
                                            input_tensor_name, classes_tensor_name,
                                            input_shapes = input_tensor_shape)
#input_shapes = input_tensor_shape

#converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

with open(out_path, "wb") as f:
    f.write(tflite_model)

