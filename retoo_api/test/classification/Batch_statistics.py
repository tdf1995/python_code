# 批量分类，统计评估指标：混淆矩阵
import os
import glob
import cv2
import numpy as np
from Classification import Classification
from segmentation_assist import *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def confusion_matrix_process(truth_labels,predict_labels,labels_list=None):
    if labels_list:
        for i in range(len(truth_labels)):
            truth_labels[i]=labels_list[int(truth_labels[i])]
            predict_labels[i] = labels_list[int(predict_labels[i])]
    cm = confusion_matrix(truth_labels, predict_labels,labels=labels_list)
    plt.imshow(cm, cmap=plt.cm.Greens)
    indices = range(len(cm))
    plt.xticks(indices, labels_list,rotation=45,verticalalignment='top')
    plt.yticks(indices, labels_list,rotation=45)
    plt.title('confusion_matrix')
    plt.colorbar()
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(y, x), horizontalalignment='center', verticalalignment='center')

    plt.ylabel('True label')  # 坐标轴标签
    plt.xlabel('Predicted label',verticalalignment='top')  # 坐标轴标签

    plt.show()
    np.savetxt('./confusion_matrix.csv', cm, delimiter=',')

if __name__ == '__main__':

    pb_path = r'F:\models-master\cell\分类\pb/frozen_inception_resnet_v2_inf_graph.pb'
    img_root_path = r'F:\models-master\cell\分类\test_set'

    clf = Classification(pb_path)

    img_paths = glob.glob(img_root_path + '/*/*.jpg')
    y_pred = []
    y_truth = []
    for img_path in img_paths:
        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        truth_cls = img_path.split(os.sep)[-2]
        height = image.shape[0]
        width = image.shape[1]
        clf.inference_detection_model(image)
        pred = clf.results_analysis()

        y_pred.append(str(pred))
        print(pred)
        y_truth.append(truth_cls)
        print(truth_cls)

    confusion_matrix_process(y_truth,y_pred,labels_list=[
                                           'danhe',
                                           'linba',
                                           'shijian',
                                           'shisuan',
                                           'zhongxing',
                                           'broken',
                                            'background'
                                           ])
