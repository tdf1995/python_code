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
    np.savetxt('./confusion_matrix.csv', cm, fmt='%d', delimiter=',')

def batch_test(clf,image_root_path,labels_list=None):
    '''
    批量测试准确率及混淆矩阵
    :param clf: 分类模型
    :param image_root_path: 图像根目录，下级目录为类别子目录
    :param label_list: 类别列表，eg.['danhe','linba']
    :return:
    '''

    img_paths = glob.glob(image_root_path + '/*/*.jpg')
    y_pred = []
    y_truth = []
    correct = 0
    for img_path in img_paths:
        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        truth_cls = img_path.split(os.sep)[-2]

        clf.inference_detection_model(image)
        pred = str(clf.results_analysis())

        y_pred.append(pred)
        y_truth.append(truth_cls)
        if pred==truth_cls:
            correct += 1

    confusion_matrix_process(y_truth,y_pred,labels_list=labels_list)
    correct_rate = float(correct)/len(y_truth)
    print("总体准确率为:",correct_rate)

def batch_clfmove(clf, img_root_path, target_root_path):
    '''
    批量将未分类的图片用模型分类并移动到相应文件夹
    :param clf: 分类模型
    :param img_root_path:根目录
    :param target_root_path:移动位置根目录
    :return:
    '''

    rootdir = os.listdir(img_root_path)
    for e in rootdir:
        subdir = os.path.join(img_root_path, e)  # 子文件及子文件夹路径
        if os.path.isfile(subdir):  # 如果是文件
            if os.path.splitext(subdir)[1] in {'.jpg', '.bmp', '.png'}:
                image = cv2.imdecode(np.fromfile(subdir, dtype=np.uint8), -1)
                clf.inference_detection_model(image)
                pred = str(clf.results_analysis())
                target_path = os.path.join(target_root_path,pred)
                if not os.path.exists(target_path):
                    os.makedirs(target_path)
                target_file_path = os.path.join(target_path, e)
                if os.path.splitext(subdir)[1] == '.jpg':
                    cv2.imencode('.jpg', image)[1].tofile(target_file_path)
                if os.path.splitext(subdir)[1] == '.bmp':
                    cv2.imencode('.bmp', image)[1].tofile(target_file_path)
                if os.path.splitext(subdir)[1] == '.png':
                    cv2.imencode('.png', image)[1].tofile(target_file_path)
        elif os.path.isdir(subdir):  # 如果是路径
            batch_clfmove(clf, subdir, target_root_path)

if __name__ == '__main__':

    # 测试用例
    pb_path = r'F:\models-master\cell\分类\pb/frozen_inception_resnet_v2_inf_graph.pb'
    img_root_path = r'F:\models-master\cell\分类\test_set'

    clf = Classification(pb_path)

    batch_test(clf, img_root_path, labels_list=[
                                           'danhe',
                                           'linba',
                                           'shijian',
                                           'shisuan',
                                           'zhongxing',
                                           'broken',
                                            'background'
                                           ])

    batch_clfmove(clf,img_root_path,r'F:\models-master\cell\分类\1')
