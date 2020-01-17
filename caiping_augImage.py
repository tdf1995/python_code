import cv2
import os
import numpy as np
import random

pic_path = r'E:\蒸浏记\分割\增强\待新增\20190927152745277.jpg'
png_path = r'E:\蒸浏记\分割\增强\待新增\20190927152745277.png'
empty_path = r'E:\蒸浏记\分割\增强\背景图\20190927151118303.jpg'

crop_path = r'E:\蒸浏记\分割\增强\小图'
aug_path = r'E:\蒸浏记\分割\增强\新增'

img = cv2.imdecode(np.fromfile(pic_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
img_mask = cv2.imdecode(np.fromfile(png_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
# empty_pic = cv2.imdecode(np.fromfile(empty_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)


_, mask_binary = cv2.threshold(img_mask, 0, 255, cv2.THRESH_BINARY)
contours,hierarchy= cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for i in range(len(contours)):
    x, y, loc_w, loc_h = cv2.boundingRect(contours[i])
    x1, y1, x2, y2 = x, y, x + loc_w, y + loc_h
    img_crop = img[y1:y2,x1:x2]
    png_crop = img_mask[y1:y2,x1:x2]

    cv2.imencode('.jpg', img_crop)[1].tofile(os.path.join(crop_path, os.path.basename(pic_path))[:-4]+'_'+str(i)+'.jpg')
    cv2.imencode('.png', png_crop)[1].tofile(os.path.join(crop_path, os.path.basename(pic_path)[:-4]+'_'+str(i)+'.png'))


    # cv2.imencode('.jpg', img_crop)[1].tofile(pic_path[:-4]+'_crop.jpg')
    # cv2.imencode('.png', png_crop)[1].tofile(png_path[:-4] + '_crop.png')

    (h, w) = img_crop.shape[:2]
    center = (w / 2, h / 2)
    for j in range(100):
        for k in range(100):
            angle = random.randint(-90,90)
            M = cv2.getRotationMatrix2D(center,angle,1.0)

            img_crop_random_rot = cv2.warpAffine(img_crop,M,(w,h),cv2.INTER_NEAREST)
            png_crop_random_rot = cv2.warpAffine(png_crop, M, (w, h),flags=cv2.INTER_NEAREST)
            # cv2.imencode('.png', png_crop_random_rot)[1].tofile(os.path.join(aug_path,
            #                                                        os.path.basename(pic_path)[:-4] + '_' + str(j) + str(
            #                                                            k) + '_' + str(angle) + '_crop'+'.png'))
            for f in range(3):
                img_crop_random_rot[..., f] = np.where(png_crop_random_rot == 0, 0, img_crop_random_rot[..., f])



            # img_crop_random_rot = img_crop
            # png_crop_random_rot = png_crop

            empty = np.zeros((2448, 3264, 3), dtype=np.uint8)
            empty_png = np.zeros((2448, 3264), dtype=np.uint8)
            empty_pic = cv2.imdecode(np.fromfile(empty_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            s_x, s_y = [405, 611]
            e_x, e_y = [2578, 2263]
            step = 200
            s_x = s_x + step * k
            s_y = s_y + step * j
            t_x, t_y= (s_x+loc_w, s_y+loc_h)
            if (t_x>e_x)or(t_y>e_y):
                break

            empty[s_y:t_y, s_x:t_x] = img_crop_random_rot
            empty_png[s_y:t_y, s_x:t_x] = png_crop_random_rot

            # cv2.imencode('.jpg', empty)[1].tofile(empty_path[:-4] + '_crop1.jpg')
            for g in range(3):
                empty_pic[..., g] = np.where(empty_png == 0, empty_pic[..., g], empty[..., g])
            cv2.imencode('.jpg', empty_pic)[1].tofile(os.path.join(aug_path,os.path.basename(pic_path)[:-4] + '_'+str(i)+'_'+str(j)+str(k)+'_'+str(angle)+'.jpg'))
            cv2.imencode('.png', empty_png)[1].tofile(os.path.join(aug_path,os.path.basename(pic_path)[:-4] + '_'+str(i)+'_'+str(j)+str(k)+'_'+str(angle)+'.png'))

        if t_y>e_y:
            break


