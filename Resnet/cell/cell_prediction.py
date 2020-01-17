import os
import shutil
import glob
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import numpy as np

def read_image(img,size):
    im = img.resize((size,size))
    x = image.img_to_array(im)
    # x = x / 255.0
    x = np.expand_dims(x,axis=0)
    return x

if __name__ == '__main__':

    model = load_model('logs/cell_9class.h5')
    train_img_path = r'E:\细胞\x100\5类_tdf'
    img_test_path = r'D:\python code\Resnet\cell\prediction'
    save_path = r'D:\python code\Resnet\cell\result'
    TRAIN_SIZE = 300
    classes = os.listdir(train_img_path)
    classes.sort()
    img_list = glob.glob(img_test_path + '/*/*')

    for i, img in enumerate(img_list):
        print(str(i) + ' / ' + str(len(img_list)))
        if 'Thumbs.db' not in img:
            img_name = os.path.basename(img)

            im = Image.open(img)
            im = read_image(im, TRAIN_SIZE)
            predict = model.predict(im)

            label_pre = np.argmax(predict)
            cls_out = classes[int(label_pre)]

            if not os.path.exists(os.path.join(save_path, cls_out)):
                os.mkdir(os.path.join(save_path, cls_out))
            shutil.copy(img, os.path.join(save_path, cls_out, img_name))