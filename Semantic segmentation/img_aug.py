import cv2
import os
import numpy as np
import glob
import math
import random
from albumentations import (VerticalFlip, HorizontalFlip, Flip, RandomRotate90, Rotate, ShiftScaleRotate, CenterCrop, OpticalDistortion, GridDistortion, ElasticTransform, HueSaturationValue,
                            RGBShift, RandomBrightnessContrast, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise, CLAHE, ChannelShuffle, InvertImg, RandomGamma, ToGray, PadIfNeeded, Compose
                           )

# Flip:翻转
# Rotate:旋转
# Brightness:亮度
# Gamma:伽马
# Elastic:弹性变换
# OpticalDistortion:光学失真
# CenterCrop:中心裁剪
# HueSaturationValue:色相饱和度
# Constract:对比度
# GridDistortion:网格扭曲
# Blur:模糊
aug_methods = [
               'Flip',
               # 'Rotate',
               'RandomBrightnessContrast',
               'Gamma',
               # 'Elastic',# 类似水纹在图像前
               # 'OpticalDistortion',
               # 'CenterCrop',
               'GridDistortion',
               'Blur',
               ]

img_root_path = r'E:\菜品\分割\训练标注数据\第三批'
mask_root_path = r'E:\菜品\分割\训练标注数据\第三批'
target_num = 2500

def Img_Aug(img, mask, img_path, mask_path, flag):
    original_height = img.shape[0]
    original_width = img.shape[1]

    # 首先进行翻转旋转
    if 'Gamma' in aug_methods:
        aug = RandomGamma(gamma_limit=(70, 130),p=0.5)

        augmented = aug(image=img, mask=mask)

        img = augmented['image']
        mask = augmented['mask']

    if 'Blur' in aug_methods:
        aug = Blur(blur_limit=7, p=0.5)

        augmented = aug(image=img, mask=mask)

        img = augmented['image']
        mask = augmented['mask']

    if 'RandomBrightnessContrast' in aug_methods:
        aug = RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1,p=0.5)

        augmented = aug(image=img, mask=mask)

        img = augmented['image']
        mask = augmented['mask']

    if 'Flip'in aug_methods:
        state = "Horizontal"
        state_to_code = {
            "Both Horizontal and Vertical": -1,
            "Vertical": 0,
            "Horizontal": 1,
        }
        aug = Flip(p=0.5)
        augmented = aug(image=img, mask=mask, d=state_to_code[state])
        img = augmented['image']
        mask = augmented['mask']

    if 'Rotate' in aug_methods:
        aug = Rotate(limit=30,p=0.5)
        augmented = aug(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']

    if 'CenterCrop' in aug_methods:
        aug = CenterCrop(p=0.5, height=int(original_height*0.85), width=int(original_width*0.85))

        augmented = aug(image=img, mask=mask)

        img = augmented['image']
        mask = augmented['mask']

    if 'Elastic' in aug_methods:
        aug = ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)

        augmented = aug(image=img, mask=mask)

        img = augmented['image']
        mask = augmented['mask']

    if 'GridDistortion' in aug_methods:
        aug = GridDistortion(num_steps=5,distort_limit=0.1,p=0.5)

        augmented = aug(image=img, mask=mask)

        img = augmented['image']
        mask = augmented['mask']

    if 'OpticalDistortion' in aug_methods:
        aug = OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5)

        augmented = aug(image=img, mask=mask)

        img = augmented['image']
        mask = augmented['mask']

    cv2.imencode('.jpg', img)[1].tofile(img_path[:-4] + '_aug_' + str(flag) + '.jpg')
    cv2.imencode('.png', mask)[1].tofile(mask_path[:-4] + '_aug_' + str(flag) + '.png')



img_files = glob.glob(img_root_path+'\*.jpg')
recent_num = len(img_files)
if recent_num >=  target_num:
    print('数量已够！')
else:
    d = math.floor(float(target_num - recent_num) / float(recent_num))
    v = target_num - (d + 1) * recent_num
    flag = random.sample(range(1, recent_num), v)
    img_flag = 0

    for img_file in img_files:
        times = d
        img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), -1)
        mask_path = os.path.join(mask_root_path, os.path.basename(img_file)[:-4]+'.png')
        try:
            mask = cv2.imdecode(np.fromfile((mask_path), dtype=np.uint8),-1)
        except:
            print('find mask error:')
            print(mask_path)
            continue
        img.astype(np.float32)
        img_flag = img_flag + 1
        if img_flag in flag: # 如果是被选中的多出来的图片，多增强一次
            times = times + 1
        for a in range(times):# 该图片要增强times次
            Img_Aug(img, mask, img_file, mask_path, a)


