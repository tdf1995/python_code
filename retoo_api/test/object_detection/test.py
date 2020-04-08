import detect_test

faster_rcnn_path = 'frozen_inference_graph.pb'
# input_node = 'image_tensor:0'
# output_node = {
#     'box':'detection_boxes:0',
#     'score':'detection_scores:0',
#     'class':'detection_classes:0'
# }
yolo_path = 'yolo_voc2012.pb'
input_node = 'input_1:0'
output_node = {
    'box':'lambda_1/concat_8:0',
    'score':'lambda_1/concat_9:0',
    'class':'lambda_1/concat_10:0'
}
# detect_test.Detect_DrawBox(pb_path=faster_rcnn_path,folder_path='test',input_node=input_node,output_node_list=output_node)
# detect_test.Detect_cropObject(pb_path=faster_rcnn_path, folder_path='test',crop_path='新建文件夹',input_node= input_node, output_node_list=output_node)
detect_test.Detect_DrawBox(pb_path=yolo_path, folder_path='E:\多目标跟踪\检测训练挑选', input_node=input_node, output_node_list=output_node)