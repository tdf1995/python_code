# 该脚本从x100细胞小图到60x小图
import os
import cv2
import glob
import numpy as np
import re
import random
from os import walk
import shutil

x100_root_path = r'E:\新建文件夹\正常样本/'
target_x60_path = r'E:\新建文件夹\x60_分类/'
all_x60_path = r'\\192.168.1.251\1项目-2细胞\0血液细胞项目\图片汇总\新样机采集图片\1104_40倍小图汇总/'


for (dirpath, dirnames,_) in walk(x100_root_path):
    for dirname in dirnames:
        sub_path = os.path.join(x100_root_path, dirname)
        x100_pics = glob.glob(sub_path+'\*.jpg')
        for x100_pic in x100_pics:
            a = re.split('[_:]', os.path.basename(x100_pic))
            if len(a)!= 10:
                continue
            flag1 = a[1]
            flag2 = a[9][:-4]
            x60_pics = os.listdir(all_x60_path)
            for x60_pic in x60_pics:
                b = re.split('[_:]', x60_pic)
                if len(b) != 10:
                    continue
                if (b[1]==flag1)and(b[9][:-4]==flag2):
                    shutil.move(os.path.join(all_x60_path, x60_pic), os.path.join(target_x60_path, dirname, x60_pic))

                    print('find one!')


# ori_crop_pics = glob.glob(ori_crop_pic_path+'\*.jpg')
# pics = os.listdir(pic_path)
# for ori_crop_pic in ori_crop_pics:
#     crop_pic = cv2.imdecode(np.fromfile(ori_crop_pic, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
#     ori_height = crop_pic.shape[0]
#     ori_width= crop_pic.shape[1]
#     a = re.split('[_:]', os.path.basename(ori_crop_pic))
#     pic_name = a[0]+'_'+a[1]+'_'+a[2]+'_'+a[3]+'_'+a[4]+'_'+a[10]+'_'+a[11]
#     if pic_name in pics:
#         print('find ',pic_name+'!')
#         pic = cv2.imdecode(np.fromfile(os.path.join(pic_path,pic_name), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
#         center_x = int(a[7])
#         center_y = int(a[8])
#         for i in range(crop_num): # 一个细胞截几次
#             Zoom_seed = random.uniform(0.75, 1.15)
#             Pan_x_sedd = random.randint(-15, 15)
#             Pan_y_sedd = random.randint(-15, 15)
#             new_xmin = center_x - ori_height/2*Zoom_seed + Pan_x_sedd
#             if new_xmin < 0:
#                 new_xmin = 0
#             new_xmax = center_x + ori_height/2*Zoom_seed + Pan_x_sedd
#             new_ymin = center_y - ori_width/2*Zoom_seed + Pan_y_sedd
#             if new_ymin < 0:
#                 new_ymin = 0
#             new_ymax = center_y + ori_width/2*Zoom_seed + Pan_y_sedd
#             new_crop_pic = pic[int(np.round(new_ymin)):int(np.round(new_ymax)),int(np.round(new_xmin)):int(np.round(new_xmax))]
#             cv2.imencode('.jpg', new_crop_pic)[1].tofile(os.path.join(new_crop_pic_path,os.path.basename(ori_crop_pic)[:-4]+'_'+str(i)+'.jpg'))
#     # print(a)
