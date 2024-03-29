# 此程序用于图像增强

import cv2
import tensorflow as tf
import numpy as np
from scipy import misc
import os
import math
import random

folder_path = r'\\192.168.1.251\1项目-2细胞\0血液细胞项目\图片汇总\新样机采集图片\3-已审核图片\0-原始数据\1106_40倍小图汇总_李康军审核\正常样本_审2\2.shijian'
target_img_num = 1999



img_num = 0

def Get_file_num(root_path):
   global img_num
   rootdir = os.listdir(root_path)
   for e in rootdir:
       subdir = os.path.join(root_path,e)   #   子文件及子文件夹路径
       if os.path.isfile(subdir):   #   如果是文件
            if os.path.splitext(subdir)[1] in {'.jpg','.bmp','.png'}:
                img_num = img_num + 1
       elif os.path.isdir(subdir):  #   如果是路径
           Get_file_num(subdir)

with tf.Session() as sess:
    shape = (None, None, 3)
    image = tf.placeholder(dtype=tf.float32, shape=shape)


    # 平移
    def random_pan_image_func(image):
        image.astype(int)
        up_down = np.random.randint(-10, 10)
        if up_down >= 0:
            up = up_down
            down = 0
        else:
            up = 0
            down = -up_down
        left_right = np.random.randint(-10, 10)
        if left_right >= 0:
            left = left_right
            right = 0
        else:
            left = 0
            right = -left_right
        (height, width, _) = image.shape
        # print("up",up)
        # print("down", down)
        # print("left", left)
        # print("right", right)
        img_pan = cv2.copyMakeBorder(image, up, down, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        img_pan = img_pan[0:height, 0:width]
        return img_pan


    img_Pan = tf.py_func(random_pan_image_func, [image], tf.float32)


    # 图像旋转（填0）
    def random_rotate_image_func(image):
        angle = np.random.uniform(low=-10.0, high=10.0)
        return misc.imrotate(image, angle, 'bicubic')


    img_rotate = tf.py_func(random_rotate_image_func, [img_Pan], tf.uint8)


    # 图像亮度，对比度调整
    img_brightness = tf.image.random_brightness(img_rotate, 0.5)
    img_contrast = tf.image.random_contrast(img_brightness, 0.7, 1.3)# 对比度范围

    # 图像水平上下翻转
    # img_up_down = tf.image.random_flip_up_down(image)
    #
    # img_left_right = tf.image.random_flip_left_right(image)

    image_hue = tf.image.random_hue(img_contrast, 0.2)

    # 高斯噪声
    noise = tf.random_normal(shape=tf.shape(image), mean=0, stddev =1, dtype=tf.float32)
    image_hue = tf.cast(image_hue, dtype=tf.float32)
    img_noise = tf.add(image_hue, noise)




    sess.run(tf.global_variables_initializer())

    Get_file_num(folder_path)
    d = math.floor(float(target_img_num-img_num)/float(img_num))
    v = target_img_num - (d+1)*img_num

    flag = random.sample(range(1, img_num), v)
    img_flag = 0


    def Img_Aug(root_path, sess):
        global img_flag
        rootdir = os.listdir(root_path)
        for e in rootdir:
            subdir = os.path.join(root_path, e)  # 子文件及子文件夹路径
            if os.path.isfile(subdir):  # 如果是文件
                if os.path.splitext(subdir)[1] in {'.jpg', '.bmp', '.png'}:
                    times = d
                    img = cv2.imdecode(np.fromfile(subdir, dtype=np.uint8), -1)
                    img.astype(np.float32)
                    img_flag = img_flag + 1
                    if img_flag in flag:
                        times = times + 1
                    for a in range(times):
                        c = np.random.randint(0, 4, 1)
                        if c == 0:
                            bri_con_img = sess.run(img_contrast, {image: img})

                            if os.path.splitext(subdir)[1] == '.jpg':
                                cv2.imencode('.jpg', bri_con_img)[1].tofile(subdir[:-4]+'_aug_bri_'+str(a)+'.jpg')
                            if os.path.splitext(subdir)[1] == '.bmp':
                                cv2.imencode('.bmp', bri_con_img)[1].tofile(subdir[:-4]+'_aug_bri_'+str(a)+'.bmp')
                            if os.path.splitext(subdir)[1] == '.png':
                                cv2.imencode('.png', bri_con_img)[1].tofile(subdir[:-4]+'_aug_bri_'+str(a)+'.png')
                        if c == 1:
                            rot_img = sess.run(img_rotate, {image: img})

                            if os.path.splitext(subdir)[1] == '.jpg':
                                cv2.imencode('.jpg', rot_img)[1].tofile(subdir[:-4]+'_aug_rot_'+str(a)+'.jpg')
                            if os.path.splitext(subdir)[1] == '.bmp':
                                cv2.imencode('.bmp', rot_img)[1].tofile(subdir[:-4]+'_aug_rot_'+str(a)+'.bmp')
                            if os.path.splitext(subdir)[1] == '.png':
                                cv2.imencode('.png', rot_img)[1].tofile(subdir[:-4]+'_aug_rot_'+str(a)+'.png')
                        # if c == 2:
                        #     noise_img = sess.run(img_noise, {image: img})
                        #
                        #     if os.path.splitext(subdir)[1] == '.jpg':
                        #         cv2.imencode('.jpg', noise_img)[1].tofile(subdir[:-4]+'_aug_noise_'+str(a)+'.jpg')
                        #     if os.path.splitext(subdir)[1] == '.bmp':
                        #         cv2.imencode('.bmp', noise_img)[1].tofile(subdir[:-4]+'_aug_noise_'+str(a)+'.bmp')
                        #     if os.path.splitext(subdir)[1] == '.png':
                        #         cv2.imencode('.png', noise_img)[1].tofile(subdir[:-4]+'_aug_noise_'+str(a)+'.png')
                        if c == 2:
                            pan_img = sess.run(img_Pan, {image: img})

                            if os.path.splitext(subdir)[1] == '.jpg':
                                cv2.imencode('.jpg', pan_img)[1].tofile(subdir[:-4]+'_aug_pan_'+str(a)+'.jpg')
                            if os.path.splitext(subdir)[1] == '.bmp':
                                cv2.imencode('.bmp', pan_img)[1].tofile(subdir[:-4]+'_aug_pan_'+str(a)+'.bmp')
                            if os.path.splitext(subdir)[1] == '.png':
                                cv2.imencode('.png', pan_img)[1].tofile(subdir[:-4]+'_aug_pan_'+str(a)+'.png')
                        # if c == 4:
                        #     bri_con_img = sess.run(img_contrast, {image: img})
                        #
                        #     if os.path.splitext(subdir)[1] == '.jpg':
                        #         cv2.imencode('.jpg', bri_con_img)[1].tofile(subdir[:-4]+'_aug_'+str(a)+'.jpg')
                        #     if os.path.splitext(subdir)[1] == '.bmp':
                        #         cv2.imencode('.bmp', bri_con_img)[1].tofile(subdir[:-4]+'_aug_'+str(a)+'.bmp')
                        #     if os.path.splitext(subdir)[1] == '.png':
                        #         cv2.imencode('.png', bri_con_img)[1].tofile(subdir[:-4]+'_aug_'+str(a)+'.png')

                        if c == 3:
                            hue_img = sess.run(image_hue, {image: img})

                            if os.path.splitext(subdir)[1] == '.jpg':
                                cv2.imencode('.jpg', hue_img)[1].tofile(subdir[:-4]+'_aug_hue_'+str(a)+'.jpg')
                            if os.path.splitext(subdir)[1] == '.bmp':
                                cv2.imencode('.bmp', hue_img)[1].tofile(subdir[:-4]+'_aug_hue_'+str(a)+'.bmp')
                            if os.path.splitext(subdir)[1] == '.png':
                                cv2.imencode('.png', hue_img)[1].tofile(subdir[:-4]+'_aug_hue_'+str(a)+'.png')
            elif os.path.isdir(subdir):  # 如果是路径
                Img_Aug(subdir,sess)
    Img_Aug(folder_path,sess)

