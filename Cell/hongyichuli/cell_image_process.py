
import os
import sys

from xml.dom import minidom
import cv2
import numpy as np
# import cv
import tensorflow as tf
import random
import shutil
import math
from imgaug import augmenters as iaa

def get_bounding_point(dom):
    points_b_list=[]
    objs = dom.getElementsByTagName("object")
    label_list=[]
    for obj in objs:
        points_b=[]
        labelNode=obj.getElementsByTagName("name")
        labelStr=labelNode[0].childNodes[0].nodeValue
        label_str_list=labelStr.split('.')
        label=int(label_str_list[0])
        label_list.append(label)
        polygen=obj.getElementsByTagName("polygen")
        pxNode=polygen[0].getElementsByTagName("points_x")
        pyNode=polygen[0].getElementsByTagName("points_y")
        px_str=pxNode[0].childNodes[0].nodeValue
        py_str =pyNode[0].childNodes[0].nodeValue

        px_=px_str.split(',')
        py_=py_str.split(',')

        for i in range(len(px_)):
            if px_[i]!='':
                points_b.append((int(float(px_[i])),int(float(py_[i]))))

        points_b_list.append(points_b)

    return points_b_list,label_list


def gen_cls_img():

    folder_path = r'E:\阜阳现场测试\分割\黄\356.黄山(金光明)'

    img_files = os.listdir(os.path.join(folder_path))
    for file in img_files:
        if file.endswith('jpg'):
            xml_file_path = os.path.join(folder_path, file[:-3] + 'xml')
            if not os.path.exists(xml_file_path):
                print(xml_file_path, 'not found!')
                continue

            pathMask = os.path.join(folder_path, file[:-4] + '.png')
            if not os.path.exists(pathMask):

                dom = minidom.parse(xml_file_path)
                pt_bounding, labels = get_bounding_point(dom)
                # img_filename = get_img_filename(dom)
                imgPath = os.path.join(folder_path, file)

                # imgOri = cv2.imread(imgPath)
                imgOri = cv2.imdecode(np.fromfile(imgPath, dtype=np.uint8), cv2.IMREAD_COLOR)
                h, w, d = imgOri.shape
                # img = np.zeros((h, w, d), dtype=np.uint8)
                # groundTruth = np.zeros((h, w, 3), dtype=np.uint16)
                groundTruth = np.zeros((h, w, 3), dtype=np.uint8)

                for n in range(len(pt_bounding)):
                    # label=dir_class_dic[str(labels[n])]
                    label = labels[n]
                    if label >= 1 and label <= 5:
                        cv2.fillPoly(groundTruth, np.array([pt_bounding[n]], np.int32), (label, label, label),
                                     cv2.LINE_8)
                    else:
                        print(xml_file_path, label)

                groundTruth = cv2.cvtColor(groundTruth, cv2.COLOR_RGB2GRAY)
                pathMask = os.path.join(folder_path, file[:-4] + '.png')
                # cv2.imwrite(pathMask, groundTruth)
                cv2.imencode('.png', groundTruth)[1].tofile(pathMask)


def random_flip_left_right_image_and_label(image, cls):
  """Randomly flip an image and label horizontally (left to right).

  Args:
    image: A 3-D tensor of shape `[height, width, channels].`
    label: A 3-D tensor of shape `[height, width, 1].`

  Returns:
    A 3-D tensor of the same type and shape as `image`.
    A 3-D tensor of the same type and shape as `label`.
  """
  uniform_random = tf.random_uniform([], 0, 1.0)
  mirror_cond = tf.less(uniform_random, .5)
  image = tf.cond(mirror_cond, lambda: tf.reverse(image, [1]), lambda: image)
  cls = tf.cond(mirror_cond, lambda: tf.reverse(cls, [1]), lambda: cls)

  return image, cls

def random_flip_up_down_image_and_label(image, cls):
  """Randomly flip an image and label horizontally (left to right).

  Args:
    image: A 3-D tensor of shape `[height, width, channels].`
    label: A 3-D tensor of shape `[height, width, 1].`

  Returns:
    A 3-D tensor of the same type and shape as `image`.
    A 3-D tensor of the same type and shape as `label`.
  """
  uniform_random = tf.random_uniform([], 0, 1.0)
  mirror_cond = tf.less(uniform_random, .5)
  image = tf.cond(mirror_cond, lambda: tf.reverse(image, [0]), lambda: image)
  cls = tf.cond(mirror_cond, lambda: tf.reverse(cls, [0]), lambda: cls)

  return image, cls

def random_flip_left_right_image_and_label(image, cls):
  """Randomly flip an image and label horizontally (left to right).

  Args:
    image: A 3-D tensor of shape `[height, width, channels].`
    label: A 3-D tensor of shape `[height, width, 1].`

  Returns:
    A 3-D tensor of the same type and shape as `image`.
    A 3-D tensor of the same type and shape as `label`.
  """
  uniform_random = tf.random_uniform([], 0, 1.0)
  mirror_cond = tf.less(uniform_random, .5)
  image = tf.cond(mirror_cond, lambda: tf.reverse(image, [1]), lambda: image)
  cls = tf.cond(mirror_cond, lambda: tf.reverse(cls, [1]), lambda: cls)

  return image, cls


def random_flip_up_down_image_and_label(image, cls):
  """Randomly flip an image and label horizontally (left to right).

  Args:
    image: A 3-D tensor of shape `[height, width, channels].`
    label: A 3-D tensor of shape `[height, width, 1].`

  Returns:
    A 3-D tensor of the same type and shape as `image`.
    A 3-D tensor of the same type and shape as `label`.
  """
  uniform_random = tf.random_uniform([], 0, 1.0)
  mirror_cond = tf.less(uniform_random, .5)
  image = tf.cond(mirror_cond, lambda: tf.reverse(image, [0]), lambda: image)
  cls = tf.cond(mirror_cond, lambda: tf.reverse(cls, [0]), lambda: cls)

  return image, cls

def random_rotate_image_and_label(image, cls,angle=1):
  """Randomly flip an image and label horizontally (left to right).

  Args:
    image: A 3-D tensor of shape `[height, width, channels].`
    label: A 3-D tensor of shape `[height, width, 1].`

  Returns:
    A 3-D tensor of the same type and shape as `image`.
    A 3-D tensor of the same type and shape as `label`.
  """
  uniform_random = tf.random_uniform([], 0, 1.0)
  mirror_cond = tf.less(uniform_random, .5)
  image = tf.cond(mirror_cond, lambda: tf.image.rot90(image,tf.cond(mirror_cond,lambda:1,lambda:3)), lambda: image)
  cls = tf.cond(mirror_cond, lambda: tf.image.rot90(cls, tf.cond(mirror_cond,lambda:1,lambda:3)), lambda: cls)

  return image, cls

# 定义旋转rotate函数
def rotate_image(image, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]

    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)

    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h),borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))

    # 返回旋转后的图像
    return rotated

def random_rotate_image_and_label_random_angle(image,cls):
  """Randomly flip an image and label horizontally (left to right).

  Args:
    image: A 3-D tensor of shape `[height, width, channels].`
    label: A 3-D tensor of shape `[height, width, 1].`

  Returns:
    A 3-D tensor of the same type and shape as `image`.
    A 3-D tensor of the same type and shape as `label`.
  """
  angle=np.random.randint(-45,45)
  image = tf.py_func(rotate_image,[image,angle],tf.uint8)
  cls = tf.py_func(rotate_image,[cls,angle],tf.uint8)

  return image,cls


def random_op_2(img):
    r=np.random.randint(0,5,1)
    if (r==0):
        imgaug=tf.image.random_brightness(img, 0.2)
        imgaug=tf.image.random_contrast(imgaug, 0.8, 1.4)
    elif (r==1):
        imgaug=tf.image.random_brightness(img, 0.2)
        imgaug=tf.image.random_hue(imgaug, 0.03)
    elif (r == 2):
        imgaug = tf.image.random_brightness(img, 0.2)
        imgaug=tf.image.random_saturation(imgaug, 0.9, 1.1)
    elif (r == 3):
        imgaug = tf.image.random_contrast(img,  0.8, 1.4)
        imgaug=tf.image.random_hue(imgaug,0.03)
    elif (r == 4):
        imgaug=tf.image.random_hue(img,0.03)
        imgaug=tf.image.random_saturation(imgaug, 0.9, 1.1)
    elif (r == 5):
        imgaug= tf.image.random_contrast(img,  0.8, 1.4)
        imgaug=tf.image.random_saturation(imgaug, 0.9, 1.1)

    return imgaug


def enhance_image_segment():

    aug = iaa.PiecewiseAffine(scale=(0.005, 0.02),nb_rows=4,nb_cols=4,random_state=1)

    sess = tf.Session()

    image = tf.placeholder(dtype=tf.uint8, shape=(None,None,3))
    cls = tf.placeholder(dtype=tf.uint8, shape=(None,None,1))

    brightness, rotate90l = random_flip_left_right_image_and_label(random_op_2(image), cls=cls)

    contrast, rotate180l = random_flip_up_down_image_and_label(random_op_2(image), cls=cls)

    hue, rotate270l = random_rotate_image_and_label(random_op_2(image), cls=cls)

    saturation, rotate360l = random_rotate_image_and_label_random_angle(random_op_2(image), cls=cls)

    sess.run(tf.initialize_all_variables())

    root_dir = r"E:\projects\semantic_segmentation_github\image_dataset\x40_Loc_polygen\0807_train_all"
    # folder_list=['0.danhe','1.linba','2.shijian','3.shisuan','4.zhongxing','5.unknown','6.notcell']
    folder_list=['0.danhe','1.linba','2.shijian','3.shisuan','4.zhongxing','6.notcell']

    normal_img_count=[10000,10000,10000,10000,10000]
    for i in range(len(folder_list)):
        folder=folder_list[i]
        folder_path=os.path.join(root_dir,folder)
        files = os.listdir(folder_path)
        img_files = []
        for file in files:
            if file.endswith('bmp'):
                img_files.append(file)

        num_of_img = len(img_files)
        loop = normal_img_count[i] - num_of_img
        if loop > 0:
            r = np.random.randint(0, num_of_img - 1, loop)
            for l in range(loop):

                c = np.random.randint(0, 3, 1)

                filename = img_files[r[l]]

                pathImg = os.path.join(folder_path, filename[:-3] + 'bmp')
                pathCls = os.path.join(folder_path, filename[:-3] + 'png')

                # img_ori = cv2.imread(pathImg, cv2.IMREAD_UNCHANGED)
                # img_cls = cv2.imread(pathCls, cv2.IMREAD_UNCHANGED)
                img_ori = cv2.imdecode(np.fromfile(pathImg, dtype=np.uint8), cv2.IMREAD_COLOR)
                img_cls = cv2.imdecode(np.fromfile(pathCls, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

                h, w, _ = img_ori.shape
                img_cls = img_cls[:, :, np.newaxis]

                if c == 0:
                    bri, ro90l = sess.run([brightness, rotate90l], {image: img_ori, cls: img_cls})
                    bri = aug.augment_image(bri)
                    ro90l = aug.augment_image(ro90l)
                    cv2.imencode('.bmp', bri)[1].tofile(pathImg[:-4] + '_aug_' + str(l) + '.bmp')
                    cv2.imencode('.png', ro90l)[1].tofile(pathCls[:-4] + '_aug_' + str(l) + '.png')
                elif c == 1:
                    con, ro180l = sess.run([contrast, rotate180l], {image: img_ori, cls: img_cls})
                    con = aug.augment_image(con)
                    ro180l = aug.augment_image(ro180l)
                    cv2.imencode('.bmp', con)[1].tofile(pathImg[:-4] + '_aug_' + str(l) + '.bmp')
                    cv2.imencode('.png', ro180l)[1].tofile(pathCls[:-4] + '_aug_' + str(l) + '.png')
                elif c == 2:
                    hu, ro270l = sess.run([hue, rotate270l], {image: img_ori, cls: img_cls})
                    hu = aug.augment_image(hu)
                    ro270l = aug.augment_image(ro270l)
                    cv2.imencode('.bmp', hu)[1].tofile(pathImg[:-4] + '_aug_' + str(l) + '.bmp')
                    cv2.imencode('.png', ro270l)[1].tofile(pathCls[:-4] + '_aug_' + str(l) + '.png')
                elif c == 3:
                    sat, ro360l = sess.run([saturation, rotate360l], {image: img_ori, cls: img_cls})
                    sat = aug.augment_image(sat)
                    ro360l = aug.augment_image(ro360l)
                    cv2.imencode('.bmp', sat)[1].tofile(pathImg[:-4] + '_aug_' + str(l) + '.bmp')
                    cv2.imencode('.png', ro360l)[1].tofile(pathCls[:-4] + '_aug_' + str(l) + '.png')

#由预测的mask图生成xml标注文件
def gen_xml_from_predict():
    xml_template=r"E:\projects\semantic_segmentation_github\image_dataset\x40_Loc_polygen\0719\x40_template.xml"
    folder = ['0.danhe', '1.linba', '2.shijian', '3.shisuan', '4.zhongxing']
    dir=r"E:\projects\semantic_segmentation_github\image_dataset\x40_Loc_polygen\0719"
    for i in range(len(folder)):
        img_files=os.listdir(os.path.join(dir,folder[i]))
        for img_file in img_files:

            print(img_file)
            path=os.path.join(dir,folder[i],img_file)
            img=cv2.imread(path,cv2.IMREAD_UNCHANGED)
            exp_image = img[np.newaxis, :]

            predictions = saved_model_predictor({'image': exp_image})
            mask = predictions['classes']

            mask=mask[0,:]
            mask=mask.astype(np.uint8)

            imgBin=cv2.threshold(mask,0,255,cv2.THRESH_BINARY)

            imgBin=imgBin[1]
            _, contours, hierarchy = cv2.findContours(imgBin,
                                                      cv2.RETR_EXTERNAL,
                                                      cv2.CHAIN_APPROX_NONE)
            if len(contours)>0:
                max_cont_index=0
                max_cont=0
                for j in range(len(contours)):
                    arclen=cv2.arcLength(contours[j],True)
                    if arclen>max_cont:
                        max_cont=arclen
                        max_cont_index=j

                #得到最大轮廓
                max_length_cont=contours[max_cont_index]

                #多边形近似
                approx = cv2.approxPolyDP(max_length_cont, 3, True)

                pt_x=""
                pt_y=""
                for k in range(len(approx)):
                    # if k%8==0: #或者平均间隔
                        pt=approx[k]
                        pt_x+=str(pt[0][0])
                        pt_y += str(pt[0][1])

                        pt_x+=","
                        pt_y +=","


                dom = minidom.parse(xml_template)

                # file_name=img_file.split('\\')
                # file_name=file_name[-1]
                filenameNode = dom.getElementsByTagName("filename")
                filenameNode[0].childNodes[0].nodeValue = img_file

                objs = dom.getElementsByTagName("object")
                obj=objs[0]
                # for obj in objs:
                labelNode = obj.getElementsByTagName("name")
                labelNode[0].childNodes[0].nodeValue = str(i)

                polygen = obj.getElementsByTagName("polygen")
                pxNode = polygen[0].getElementsByTagName("points_x")
                pyNode = polygen[0].getElementsByTagName("points_y")
                pxNode[0].childNodes[0].nodeValue = pt_x
                pyNode[0].childNodes[0].nodeValue = pt_y

                savePathXml=os.path.join(dir,folder[i],img_file[:-3]+'xml')
                f = open(savePathXml[:-3] + 'xml', 'wb')
                f.write(dom.toprettyxml(encoding='utf-8'))
                f.close()

