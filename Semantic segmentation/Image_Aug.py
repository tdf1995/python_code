#!usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
import os
from xml.dom import minidom
import cv2
import numpy as np
from imgaug import augmenters as iaa
import glob

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

# def random_flip_left_right_image_and_label(image, cls):
#   """Randomly flip an image and label horizontally (left to right).
#
#   Args:
#     image: A 3-D tensor of shape `[height, width, channels].`
#     label: A 3-D tensor of shape `[height, width, 1].`
#
#   Returns:
#     A 3-D tensor of the same type and shape as `image`.
#     A 3-D tensor of the same type and shape as `label`.
#   """
#   uniform_random = tf.random_uniform([], 0, 1.0)
#   mirror_cond = tf.less(uniform_random, .5)
#   image = tf.cond(mirror_cond, lambda: tf.reverse(image, [1]), lambda: image)
#   cls = tf.cond(mirror_cond, lambda: tf.reverse(cls, [1]), lambda: cls)
#
#   return image, cls
#
#
# def random_flip_up_down_image_and_label(image, cls):
#   """Randomly flip an image and label horizontally (left to right).
#
#   Args:
#     image: A 3-D tensor of shape `[height, width, channels].`
#     label: A 3-D tensor of shape `[height, width, 1].`
#
#   Returns:
#     A 3-D tensor of the same type and shape as `image`.
#     A 3-D tensor of the same type and shape as `label`.
#   """
#   uniform_random = tf.random_uniform([], 0, 1.0)
#   mirror_cond = tf.less(uniform_random, .5)
#   image = tf.cond(mirror_cond, lambda: tf.reverse(image, [0]), lambda: image)
#   cls = tf.cond(mirror_cond, lambda: tf.reverse(cls, [0]), lambda: cls)
#
#   return image, cls

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


def rotate_image(image, angle, center=None, scale=1.0):

    (h, w) = image.shape[:2]


    if center is None:
        center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h),borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0),flags=cv2.INTER_NEAREST)


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
    aug = iaa.PiecewiseAffine(scale=(0.0005, 0.005),nb_rows=4,nb_cols=4,random_state=1)

    sess = tf.Session()

    image = tf.placeholder(dtype=tf.uint8, shape=(None,None,3))
    cls = tf.placeholder(dtype=tf.uint8, shape=(None,None,1))

    brightness, rotate90l = random_flip_left_right_image_and_label(random_op_2(image), cls=cls)

    contrast, rotate180l = random_flip_up_down_image_and_label(random_op_2(image), cls=cls)

    hue, rotate270l = random_rotate_image_and_label(random_op_2(image), cls=cls)

    saturation, rotate360l = random_rotate_image_and_label_random_angle(random_op_2(image), cls=cls)

    sess.run(tf.initialize_all_variables())

    root_dir = r"C:\Users\tdf\Desktop\error"

    # folder_list=['0.notcell','1.danhe','2.linba','3.shijian','4.shisuan','5.zhongxing','6.broken']
    folder_list=['2']

    # normal_img_count=[6500,6500,6500,6500,6500,6500,6500,]
    normal_img_count = [1000, ]
    for i in range(len(folder_list)):
        folder=folder_list[i]
        folder_path=os.path.join(root_dir,folder)
        files = os.listdir(folder_path)
        img_files = []
        for file in files:
            if file.endswith('jpg'):
                img_files.append(file)

        num_of_img = len(img_files)
        loop = normal_img_count[i] - num_of_img
        if loop > 0:
            r = np.random.randint(0, num_of_img - 1, loop)
            for l in range(loop):

                c = np.random.randint(0, 3, 1)

                filename = img_files[r[l]]
                print(filename)

                pathImg = os.path.join(folder_path, filename[:-3] + 'jpg')
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
                    cv2.imencode('.jpg', bri)[1].tofile(pathImg[:-4] + '_aug_' + str(l) + '.jpg')
                    cv2.imencode('.png', ro90l)[1].tofile(pathCls[:-4] + '_aug_' + str(l) + '.png')
                elif c == 1:
                    con, ro180l = sess.run([contrast, rotate180l], {image: img_ori, cls: img_cls})
                    con = aug.augment_image(con)
                    ro180l = aug.augment_image(ro180l)
                    cv2.imencode('.jpg', con)[1].tofile(pathImg[:-4] + '_aug_' + str(l) + '.jpg')
                    cv2.imencode('.png', ro180l)[1].tofile(pathCls[:-4] + '_aug_' + str(l) + '.png')
                elif c == 2:
                    hu, ro270l = sess.run([hue, rotate270l], {image: img_ori, cls: img_cls})
                    hu = aug.augment_image(hu)
                    ro270l = aug.augment_image(ro270l)
                    cv2.imencode('.jpg', hu)[1].tofile(pathImg[:-4] + '_aug_' + str(l) + '.jpg')
                    cv2.imencode('.png', ro270l)[1].tofile(pathCls[:-4] + '_aug_' + str(l) + '.png')
                elif c == 3:
                    sat, ro360l = sess.run([saturation, rotate360l], {image: img_ori, cls: img_cls})
                    sat = aug.augment_image(sat)
                    ro360l = aug.augment_image(ro360l)
                    cv2.imencode('.jpg', sat)[1].tofile(pathImg[:-4] + '_aug_' + str(l) + '.jpg')
                    cv2.imencode('.png', ro360l)[1].tofile(pathCls[:-4] + '_aug_' + str(l) + '.png')
        png_list = glob.glob(folder_path+'\*.png')
        gray = int(folder[:1])
        for png_file in png_list:
            png_pic = cv2.imdecode(np.fromfile(png_file, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            png_pic = cv2.threshold(png_pic, 0, 1, cv2.THRESH_BINARY)
            cv2.imencode('.png', png_pic[1])[1].tofile(png_file)




if __name__ == '__main__':
    enhance_image_segment()