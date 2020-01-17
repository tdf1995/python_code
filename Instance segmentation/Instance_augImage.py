import cv2
import os
import numpy as np
import random
import glob

BG_path = r'E:\黄豆\增强\背景\裁剪/'
# png_path = r'E:\黄豆\增强\待新增'
# jpg_path = r'E:\黄豆\增强\待新增'

crop_path = r'E:\黄豆\增强\小图/'
aug_path = r'E:\黄豆\增强\新增/'

# img = cv2.imdecode(np.fromfile(pic_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
# img_mask = cv2.imdecode(np.fromfile(png_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
# empty_pic = cv2.imdecode(np.fromfile(empty_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
BG_files = glob.glob(BG_path+'*.jpg')
crop_files = glob.glob(crop_path+'*.jpg')
crop_png_files = glob.glob(crop_path+'*.png')
rand = random.randint(0,5000)
for BG_file in BG_files:
    BG_pic = cv2.imdecode(np.fromfile(BG_file, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    empty = np.zeros(BG_pic.shape[:3], dtype=np.uint8)
    empty_png = np.zeros(BG_pic.shape[:2], dtype=np.uint8)
    BG_height, BG_width = BG_pic.shape[:2]

    seed_num = random.randint(10,15) # 随机选择一张图中种子数量
    chosen_crop_pic = random.sample(crop_files, seed_num)

    crop_pt = []
    # if seed_num > len(crop_files):
    for i in range(seed_num):
        # if i >= len(crop_files):
        #     i = i - len(crop_files)
        crop_file = chosen_crop_pic[i]
        crop_png_file = crop_file[:-4]+'.png'
        crop_pic = cv2.imdecode(np.fromfile(crop_file, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        crop_height, crop_width = crop_pic.shape[:2]
        crop_png_pic = cv2.imdecode(np.fromfile(crop_png_file, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        _, crop_png_pic = cv2.threshold(crop_png_pic, 0, i+1, cv2.THRESH_BINARY)

        if (crop_pt !=[]):

            flag = 0
            while (flag == 0):# 满足就一直循环
                flag = 1
                random_xs = random.randint(0, BG_height-crop_height)
                random_ys = random.randint(0, BG_width-crop_width)

                random_xe = random_xs + crop_height
                random_ye = random_ys + crop_width
                for j in range(len(crop_pt)):# 与之前添加的所有框，计算IOU
                    xx1 = np.maximum(random_xs, crop_pt[j][0][1])#   重叠区域(如果存在)的x的最小值
                    yy1 = np.maximum(random_ys, crop_pt[j][0][0])#   重叠区域(如果存在)的y的最小值
                    xx2 = np.minimum(random_xe, crop_pt[j][1][1])
                    yy2 = np.minimum(random_ye, crop_pt[j][1][0])

                    # areas1 = (random_xe - random_xs) * (random_ye - random_ys)
                    # areas2 = (crop_pt[j][1][0] - crop_pt[j][0][0]) * (crop_pt[j][1][1] - crop_pt[j][0][1])

                    w = np.maximum(0.0, xx2 - xx1 + 1)  # 重叠区域宽
                    h = np.maximum(0.0, yy2 - yy1 + 1)  # 重叠区域高
                    inter = int(w * h)
                    if inter != 0:# 如果相交，跳出重新random
                        flag = 0
                        break


            crop_pt.append(([random_ys, random_xs],[random_ye, random_xe]))
            for f in range(3):
                crop_pic[..., f] = np.where(crop_png_pic == 0, 0, crop_pic[..., f])
            empty[random_xs:random_xe, random_ys:random_ye] = crop_pic
            empty_png[random_xs:random_xe, random_ys:random_ye] = crop_png_pic
            for h1 in range(3):
                BG_pic[..., h1] = np.where(empty_png == 0, BG_pic[..., h1], empty[..., h1])

            # cv2.imencode('.jpg', BG_pic)[1].tofile(os.path.join(aug_path,
            #                                                     os.path.basename(BG_file)[:-4] + '_' + str(
            #                                                         seed_num) + '.jpg'))

        else:
            random_xs = random.randint(0, BG_height - crop_height)
            random_ys = random.randint(0, BG_width - crop_width)

            random_xe = random_xs + crop_height
            random_ye = random_ys + crop_width
            crop_pt.append(([random_ys, random_xs],[random_ye, random_xe]))

            for k in range(3):
                crop_pic[..., k] = np.where(crop_png_pic == 0, 0, crop_pic[..., k])
            empty[random_xs:random_xe, random_ys:random_ye] = crop_pic
            empty_png[random_xs:random_xe, random_ys:random_ye] = crop_png_pic
            for g in range(3):
                BG_pic[..., g] = np.where(empty_png == 0, BG_pic[..., g], empty[..., g])
            # cv2.imencode('.jpg', BG_pic)[1].tofile(os.path.join(aug_path,
            #                                                     os.path.basename(BG_file)[:-4] + '_' + str(
            #                                                         seed_num) + '.jpg'))


    cv2.imencode('.jpg', BG_pic)[1].tofile(os.path.join(aug_path,
                                                           os.path.basename(BG_file)[:-4]  +  '_' + str(seed_num)+'_'+str(rand) + '.jpg'))
    cv2.imencode('.png', empty_png)[1].tofile(os.path.join(aug_path,
                                                           os.path.basename(BG_file)[:-4] +  '_' + str(seed_num)+'_'+str(rand) + '.png'))



#
#
#
# _, mask_binary = cv2.threshold(img_mask, 0, 255, cv2.THRESH_BINARY)
# contours,hierarchy= cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# for i in range(len(contours)):
#     x, y, loc_w, loc_h = cv2.boundingRect(contours[i])
#     x1, y1, x2, y2 = x, y, x + loc_w, y + loc_h
#     img_crop = img[y1:y2,x1:x2]
#     png_crop = img_mask[y1:y2,x1:x2]
#
#     cv2.imencode('.jpg', img_crop)[1].tofile(os.path.join(crop_path, os.path.basename(pic_path)))
#     cv2.imencode('.png', png_crop)[1].tofile(os.path.join(crop_path, os.path.basename(pic_path)[:-4]+'.png'))
#
#
#     # cv2.imencode('.jpg', img_crop)[1].tofile(pic_path[:-4]+'_crop.jpg')
#     # cv2.imencode('.png', png_crop)[1].tofile(png_path[:-4] + '_crop.png')
#
#     (h, w) = img_crop.shape[:2]
#     center = (w / 2, h / 2)
#     for j in range(100):
#         for k in range(100):
#             angle = random.randint(-90,90)
#             M = cv2.getRotationMatrix2D(center,angle,1.0)
#
#             img_crop_random_rot = cv2.warpAffine(img_crop,M,(w,h),cv2.INTER_NEAREST)
#             png_crop_random_rot = cv2.warpAffine(png_crop, M, (w, h),flags=cv2.INTER_NEAREST)
#             # cv2.imencode('.png', png_crop_random_rot)[1].tofile(os.path.join(aug_path,
#             #                                                        os.path.basename(pic_path)[:-4] + '_' + str(j) + str(
#             #                                                            k) + '_' + str(angle) + '_crop'+'.png'))
#             for f in range(3):
#                 img_crop_random_rot[..., f] = np.where(png_crop_random_rot == 0, 0, img_crop_random_rot[..., f])
#
#
#
#             # img_crop_random_rot = img_crop
#             # png_crop_random_rot = png_crop
#
#             empty = np.zeros((2448, 3264, 3), dtype=np.uint8)
#             empty_png = np.zeros((2448, 3264), dtype=np.uint8)
#             empty_pic = cv2.imdecode(np.fromfile(empty_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
#             s_x, s_y = [488, 438]
#             e_x, e_y = [2760, 2108]
#             step = 15
#             s_x = s_x + step * k
#             s_y = s_y + step * j
#             t_x, t_y= (s_x+loc_w, s_y+loc_h)
#             if t_x>e_x:
#                 break
#
#             empty[s_y:t_y, s_x:t_x] = img_crop_random_rot
#             empty_png[s_y:t_y, s_x:t_x] = png_crop_random_rot
#
#             # cv2.imencode('.jpg', empty)[1].tofile(empty_path[:-4] + '_crop1.jpg')
#             for g in range(3):
#                 empty_pic[..., g] = np.where(empty_png == 0, empty_pic[..., g], empty[..., g])
#             cv2.imencode('.jpg', empty_pic)[1].tofile(os.path.join(aug_path,os.path.basename(pic_path)[:-4] + '_'+str(i)+'_'+str(j)+str(k)+'_'+str(angle)+'.jpg'))
#             cv2.imencode('.png', empty_png)[1].tofile(os.path.join(aug_path,os.path.basename(pic_path)[:-4] + '_'+str(i)+'_'+str(j)+str(k)+'_'+str(angle)+'.png'))
#
#         if t_y>e_y:
#             break
#
#
