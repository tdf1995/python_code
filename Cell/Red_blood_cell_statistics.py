import cv2
import os
import glob
import numpy as np
import xlwt
import xlrd
from xlutils.copy import copy
import re
import math


pic_root_path = r'E:\细胞\hongxibao\test\0'
mask_root_path = r'E:\细胞\hongxibao\test\0'
xls_path = r'.\Red_blood_cell_statisitcs.xls'
if os.path.exists(xls_path):
    cell = xlrd.open_workbook(xls_path)
    cell = copy(cell)
else:
    cell = xlwt.Workbook(encoding = 'utf-8', style_compression=0)
cellsheet = cell.add_sheet(os.path.basename(pic_root_path))
cellsheet.write(0,0,'file_name')
cellsheet.write(0,1,'面积')
cellsheet.write(0,2,'周长')
cellsheet.write(0,3,'长宽比')
cellsheet.write(0,4,'似圆度')
cellsheet.write(0,5,'凸性')
cellsheet.write(0,6,'半径')
cellsheet.write(0,7,'平滑度')
cellsheet.write(0,8,'h1')
cellsheet.write(0,9,'h2')
cellsheet.write(0,10,'h3')
cellsheet.write(0,11,'h4')
cellsheet.write(0,12,'h5')
cellsheet.write(0,13,'h6')
cellsheet.write(0,14,'h7')


pic_files = glob.glob(pic_root_path+'/*.jpg')
j = 0
for pic_file in pic_files:
    img = cv2.imdecode(np.fromfile(pic_file,np.uint8),-1)
    mask_path = os.path.join(mask_root_path, os.path.basename(pic_file)[:-4]+'.png')
    # mask = cv2.imdecode(np.fromfile(mask_path,np.uint8),-1)
    if (img is None):
        continue
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    img_mean = cv2.blur(img, (9, 9))

    img_HSV = cv2.cvtColor(img_mean, cv2.COLOR_BGR2HSV)
    vec_img_hsv = cv2.split(img_HSV)
    image_H = vec_img_hsv[0]
    image_S = vec_img_hsv[1]
    image_V = vec_img_hsv[2]

    _,img_binary = cv2.threshold(image_V, 0, 255, cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        img_binary_d = np.zeros(img_binary.shape).astype(np.uint8)
        area = cv2.contourArea(contours[i])
        Rect_box = cv2.boundingRect(contours[i])
        if Rect_box[0]<0 or Rect_box[1]<0:
            continue
        if 600 < area < 2500:
            continue
        if 8000 < area < 25000:
            # 此为红细胞
            convexHullPoints = cv2.convexHull(contours[i])
            convexHullArea = cv2.contourArea(convexHullPoints)
            dConvexity = area / convexHullArea
            if ((area < 18000 and dConvexity > 0.9) or dConvexity > 0.95):
                cv2.drawContours(img_binary_d, contours, i, (255, 255, 255), thickness=-1)
                # cv2.imshow('1',img_binary_d)
                # cv2.waitKey(0)
                contours_d, hierarchy_d = cv2.findContours(img_binary_d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                area_d = cv2.contourArea(contours[i])
                arcLen = cv2.arcLength(contours[i], True)
                arcLenHull = cv2.arcLength(convexHullPoints, True) if len(convexHullPoints)>0 else arcLen
                circularity = 4 * math.pi * area / (arcLen * arcLen)
                equi_radius = np.sqrt(area / math.pi)
                smoothness = arcLen / arcLenHull

                rotatedRect = cv2.minAreaRect(contours[i])

                aspectio = (rotatedRect[1][0]/rotatedRect[1][1]) if  rotatedRect[1][0]>rotatedRect[1][1] else rotatedRect[1][1]/rotatedRect[1][0]
                # moments = cv2.moments(img_gray)
                # humoments = cv2.HuMoments(moments)
                # humoments = np.log(np.abs(humoments))
                # print(humoments)

                # hog = cv2.HOGDescriptor((64,64),(16,16),(8,8),(8,8),9,1,4,0,2e-01,0,64)
                # hog_feature = hog.compute(img,(8,8),(8,8),((10,20),(30,30),(50,50),(70,70),(90,90),(110,110),(130,130),(150,150),(170,170),(190,190)))

                j = j + 1
                cellsheet.write(j, 0, os.path.basename(pic_file)[:-4])#文件名
                cellsheet.write(j, 1, area_d)# 面积
                cellsheet.write(j, 2, arcLen)  # 周长
                # cellsheet.write(j, 3, aspectio)  # 长宽比
                # cellsheet.write(j, 4, circularity)  # 似圆度
                # cellsheet.write(j, 5, dConvexity)  # 凸性
                # cellsheet.write(j, 6, equi_radius)  # 半径
                # cellsheet.write(j, 7, smoothness)  # 平滑度
                # cellsheet.write(j, 8, humoments[0][0])  # Hu不变矩
                # cellsheet.write(j, 9, humoments[1][0])
                # cellsheet.write(j, 10, humoments[2][0])
                # cellsheet.write(j, 11, humoments[3][0])
                # cellsheet.write(j, 12, humoments[4][0])
                # cellsheet.write(j, 13, humoments[5][0])
                # cellsheet.write(j, 14, humoments[6][0])
                print(pic_file)
    cell.save(xls_path)
