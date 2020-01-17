import os
from keras import Input
import glob
from keras.models import load_model,Model
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D,Dense
from PIL import Image
import numpy as np
from classification_models.keras import Classifiers
from collections import Counter
import csv
from keras.preprocessing.image import ImageDataGenerator
import shutil
from sklearn.metrics import confusion_matrix

def read_image(img,size):
    im = img.resize((size,size))
    x = image.img_to_array(im)
    x = x / 255.0
    x = np.expand_dims(x,axis=0)
    return x

if __name__ == '__main__':

    TRAIN_SIZE = 300
    model_path = 'logs/cell_9class.h5'
    model = load_model(model_path)
    model.summary()

    train_img_path = r'E:\细胞\x100\test1211\pic_all\x100'
    img_test_path = r'E:\细胞\x100\test1211\pic_all\x100'
    error_img_path = r'D:\python code\Resnet\cell\error'
    save_path = 'count'	

    classes = os.listdir(train_img_path)
    classes.sort()

    counter_all = Counter()
    counter_wrong = Counter()
    count_wrong_sum = 0
    count_right_sum = 0


    img_list = glob.glob(img_test_path + '/*/*')

    y_pred = []
    y_true = []
    for i, img in enumerate(img_list):
        print(str(i) + ' / ' + str(len(img_list)))
        if 'Thumbs.db' not in img:
            
            true_cls = img.split(os.sep)[-2]
            im = Image.open(img)
            im = read_image(im, TRAIN_SIZE)
            predict = model.predict(im)

            label_pre = np.argmax(predict)
            pre_cls = classes[int(label_pre)]

            y_pred.append(pre_cls)
            y_true.append(true_cls)
            counter_all[true_cls] += 1
            
            if true_cls != pre_cls:
                counter_wrong[true_cls] += 1
                count_wrong_sum+=1
                error_path = os.path.join(error_img_path, true_cls)
                if not os.path.exists(error_path):
                    os.mkdir(error_path)
                shutil.copy(img, os.path.join(error_path, os.path.basename(img)))
            else:
                count_right_sum+=1

        print('识别正确总数: ' + str(count_right_sum))
        print('识别错误总数: ' + str(count_wrong_sum))

    cm = confusion_matrix(y_true,y_pred,labels=['0.background',
                                           '1.danhe',
                                           '2.linba',
                                           '3.shijian',
                                           '4.shisuan',
                                           '5.zhongxing',
                                           '6.hongxibao',
                                           '7.xuexiaoban',
                                           '8.broken',])
    np.savetxt(save_path + '/confusion_matrix.csv',cm,delimiter = ',')
    for cls in counter_all.keys():
        with open(save_path + '/CELL_Count.csv', 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([cls, counter_all[cls], counter_wrong[cls]])
