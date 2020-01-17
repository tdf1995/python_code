# 该脚本从细胞小图搜索到原图，再通过平移和缩放获得多张小图
import os
import cv2
import glob
import numpy as np
import re
import random

ori_crop_pic_path = r'\\192.168.1.251\1项目-2细胞\0血液细胞项目\图片汇总\新样机采集图片\1023_100倍小图汇总\挑选\2.shijian'
pic_path = r'\\192.168.1.251\1项目-2细胞\0血液细胞项目\图片汇总\新样机采集图片\1028_100倍大图汇总\x100_Pic\x100_Pic\Pic'
new_crop_pic_path = r'\\192.168.1.251\1项目-2细胞\0血液细胞项目\图片汇总\新样机采集图片\1023_100倍小图汇总\挑选\新建文件夹'
crop_num = 10
ori_crop_pics = glob.glob(ori_crop_pic_path+'\*.jpg')
pics = os.listdir(pic_path)
for ori_crop_pic in ori_crop_pics:
    crop_pic = cv2.imdecode(np.fromfile(ori_crop_pic, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    ori_height = crop_pic.shape[0]
    ori_width= crop_pic.shape[1]
    a = re.split('[_:]', os.path.basename(ori_crop_pic))
    pic_name = a[0]+'_'+a[1]+'_'+a[2]+'_'+a[3]+'_'+a[4]+'_'+a[10]+'_'+a[11]
    if pic_name in pics:
        print('find ',pic_name+'!')
        pic = cv2.imdecode(np.fromfile(os.path.join(pic_path,pic_name), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        center_x = int(a[7])
        center_y = int(a[8])
        for i in range(crop_num): # 一个细胞截几次
            Zoom_seed = random.uniform(0.75, 1.15)
            Pan_x_sedd = random.randint(-15, 15)
            Pan_y_sedd = random.randint(-15, 15)
            new_xmin = center_x - ori_height/2*Zoom_seed + Pan_x_sedd
            if new_xmin < 0:
                new_xmin = 0
            new_xmax = center_x + ori_height/2*Zoom_seed + Pan_x_sedd
            new_ymin = center_y - ori_width/2*Zoom_seed + Pan_y_sedd
            if new_ymin < 0:
                new_ymin = 0
            new_ymax = center_y + ori_width/2*Zoom_seed + Pan_y_sedd
            new_crop_pic = pic[int(np.round(new_ymin)):int(np.round(new_ymax)),int(np.round(new_xmin)):int(np.round(new_xmax))]
            cv2.imencode('.jpg', new_crop_pic)[1].tofile(os.path.join(new_crop_pic_path,os.path.basename(ori_crop_pic)[:-4]+'_'+str(i)+'.jpg'))
    # print(a)
