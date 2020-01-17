from keras.applications import ResNet50
from keras.utils import np_utils, conv_utils
from keras.backend.common import normalize_data_format
from keras import layers,models
from keras.layers import Flatten, Dense, Input
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import os
import cv2

def eachFile(filepath):                 #将目录内的文件名放入列表中
    pathDir =  os.listdir(filepath)
    out = []
    for allDir in pathDir:
        child = allDir.encode('gbk')    # .decode('gbk')是解决中文显示乱码问题
        child = child.decode('gbk')
        out.append(child)
    return out

#从文件夹中获取图像数据
def get_data(train_left=0.0, train_right=0.7, train_all=0.7, resize=True, data_format=None,
             t=''):  # 从文件夹中获取图像数据
    # file_name = os.path.join(pic_dir_out, data_name + t + '_' + str(train_left) + '_' + str(train_right) + '_' + str(
    #     Width) + "X" + str(Height) + ".h5")
    #
    # if os.path.exists(file_name):  # 判断之前是否有存到文件中
    #     f = h5py.File(file_name, 'r')
    #     if t == 'train':
    #         X_train = f['X_train'][:]
    #         y_train = f['y_train'][:]
    #         f.close()
    #         return (X_train, y_train)
    #     elif t == 'test':
    #         X_test = f['X_test'][:]
    #         y_test = f['y_test'][:]
    #         f.close()
    #         return (X_test, y_test)
    #     else:
    #         return
    data_format = normalize_data_format(data_format)
    pic_dir_set = eachFile(pic_dir)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    label = 0
    for pic_dir_name in pic_dir_set:

        if not os.path.isdir(os.path.join(pic_dir, pic_dir_name)):# 如果不是文件夹，跳过
            continue
        pic_set = eachFile(os.path.join(pic_dir, pic_dir_name))# 图片列表
        pic_index = 0
        train_count = int(len(pic_set) * train_all) # 训练样本数目
        train_l = int(len(pic_set) * train_left)
        train_r = int(len(pic_set) * train_right)
        for pic_name in pic_set:
            if not os.path.isfile(os.path.join(pic_dir, pic_dir_name, pic_name)):
                continue
            img = cv2.imread(os.path.join(pic_dir, pic_dir_name, pic_name))
            if img is None:
                continue
            if (resize):
                img = cv2.resize(img, (Width, Height))
                img = img.reshape(-1, Width, Height, 3)
            if (pic_index < train_count):# 统计数量
                if t == 'train':
                    if (pic_index >= train_l and pic_index < train_r):
                        X_train.append(img)
                        y_train.append(label)
            else:
                if t == 'test':
                    X_test.append(img)
                    y_test.append(label)
            pic_index += 1
        if not len(pic_set) == 0:
            label += 1

    # f = h5py.File(file_name, 'w')
    return (X_train, y_train)
    # if t == 'train':
    #     X_train = np.concatenate(X_train, axis=0)
    #     y_train = np.array(y_train)
    #     f.create_dataset('X_train', data=X_train)
    #     f.create_dataset('y_train', data=y_train)
    #     f.close()
    #     return (X_train, y_train)
    # elif t == 'test':
    #     X_test = np.concatenate(X_test, axis=0)
    #     y_test = np.array(y_test)
    #     f.create_dataset('X_test', data=X_test)
    #     f.create_dataset('y_test', data=y_test)
    #     f.close()
    #     return (X_test, y_test)
    # else:
    #     return

def main():
    global Width, Height, pic_dir
###### 超参数设置#####################################
    Width = 280 # 设置训练集图像宽高
    Height = 280
    num_classes = 5 # 类别设置
    pic_dir = r'E:\细胞\x100\5类_tdf/'  # 数据集的路径

######################################################
    (x_train, y_train) = get_data(0.0,0.7,data_format='channels_last',t='train')
    y_train = np_utils.to_categorical(y_train, num_classes)
    input_tensor = Input(shape=(280, 280, 3))

    model = ResNet50(include_top=False,
                     weights='imagenet',
                     input_shape=(280,280,3),
                     pooling='max',
                     )
    x = model.output
    x = layers.Dense(5, activation='softmax')(x)
    model = models.Model(inputs=model.input, outputs=x)
    for layer in model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='categorical_crossentropy')
    for i, layer in enumerate(model.layers):
       print(i, layer.name)
    # model.summary()
if __name__ == '__main__':
    main()