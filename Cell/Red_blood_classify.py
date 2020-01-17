import cv2
import os
import glob
import numpy as np
import xlwt
import xlrd
from xlutils.copy import copy
import re
import math
import pickle
from sklearn import preprocessing
import shutil
import time

pic_root_path = r'\\192.168.1.251\1项目-2细胞\0血液细胞项目\图片汇总\红细胞相关\1107_100倍红细胞小图\Pic/'
# pic_root_path = r'E:\细胞\hongxibao\混合/'

pic_files = glob.glob(pic_root_path+'/*.jpg')
j = 0
saved_features = []
saved_labels = []

svm = cv2.ml.SVM_load(r"E:\hongxibao/trained_svm.xml")
# train_features = pickle.load(open(r'E:\细胞\hongxibao\混合/features.pkl', 'rb'))  # (5384,1536)
max_txt = open(r'C:\Users\tdf\Desktop\max.txt','r')
min_txt = open(r'C:\Users\tdf\Desktop\min.txt','r')
# train_features = np.stack(train_features)
# train_features =  preprocessing.normalize(train_features.T, norm='l2').T
# scalar = preprocessing.MinMaxScaler()
# scalar.fit(train_features)

max_file = max_txt.readlines()
a = re.split('[, :]',max_file[0])
min_file = min_txt.readlines()
b = re.split('[, :]', min_file[0])
for (dirpath, dirnames, _) in os.walk(pic_root_path):
    for dirname in dirnames:
        for (_, _, filenames) in os.walk(pic_root_path + dirname):
            for image_id in filenames:
                if not image_id[-4:] in {'.jpg'}:
                    continue
                dir = os.path.join(pic_root_path, dirname, image_id)
                img = cv2.imdecode(np.fromfile(dir, dtype=np.uint8), -1)
                time_start = time.time()
                # mask_path = dir[:-4]+'.png'
                # mask = cv2.imdecode(np.fromfile(mask_path,np.uint8),-1)

                if img is None:
                    continue
                img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

                img_mean = cv2.blur(img, (9, 9))

                img_HSV = cv2.cvtColor(img_mean, cv2.COLOR_BGR2HSV)
                vec_img_hsv = cv2.split(img_HSV)
                image_H = vec_img_hsv[0]
                image_S = vec_img_hsv[1]
                image_V = vec_img_hsv[2]

                _,img_binary = cv2.threshold(image_V, 0, 255, cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
                # cv2.imshow('1',img_binary)
                # cv2.waitKey(0)
                contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for i in range(len(contours)):
                    img_binary_d = np.zeros(img_binary.shape).astype(np.uint8)
                    cv2.drawContours(img_binary_d, contours, i, (255, 255, 255), thickness=-1)

                    area = cv2.contourArea(contours[i])
                    Rect_box = cv2.boundingRect(contours[i])
                    if Rect_box[0]<0 or Rect_box[1]<0:
                        continue
                    if 600 < area < 2500:
                        continue
                    if 8000 < area < 25000:
                        # 此为红细胞
                        convexHullPoints = cv2.convexHull(contours[i])
                        Hull = cv2.convexHull(contours[i], returnPoints=False)
                        convexHullArea = cv2.contourArea(convexHullPoints)
                        defects = cv2.convexityDefects(contours[i], Hull)
                        defects_num = defects.shape[0]
                        dConvexity = area / convexHullArea
                        if ((area < 18000 and dConvexity > 0.9) or dConvexity > 0.95):

                            # cv2.imshow('1',img_binary_d)
                            # cv2.waitKey(0)
                            arcLen = cv2.arcLength(contours[i], True)
                            arcLenHull = cv2.arcLength(convexHullPoints, True) if len(convexHullPoints)>0 else arcLen
                            circularity = 4 * math.pi * area / (arcLen * arcLen)
                            equi_radius = np.sqrt(area / math.pi)
                            smoothness = arcLen / arcLenHull

                            rotatedRect = cv2.minAreaRect(contours[i])

                            aspectio = (rotatedRect[1][0]/rotatedRect[1][1]) if  rotatedRect[1][0]>rotatedRect[1][1] else rotatedRect[1][1]/rotatedRect[1][0]
                            moments = cv2.moments(contours[i],False)
                            humoments = cv2.HuMoments(moments)
                            humoments = np.log(np.abs(humoments))
                            # print(humoments)

                            ImageSize = (128, 128)
                            winSize = (128, 128)

                            blockSize = (64, 64)
                            blockStride = (32, 32)
                            cellSize = (16, 16)
                            nbins = 9
                            # blockSize必须被cellSize整除
                            # winSize-blockSize必须被blockStride整除
                            # 维度=nbins*4*(winSize.height/blockStride.height-1)*(winSize.width/blockStride.width-1)
                            # img_hog = img_gray
                            img_hog = cv2.resize(img_mean, ImageSize)

                            hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
                            img_hog = hog.compute(img_hog)

                            img_hog = np.squeeze(img_hog, -1)

                            h_mean, h_std = np.squeeze(cv2.meanStdDev(image_H, img_binary_d), -1)  # 一阶颜色矩
                            s_mean, s_std = np.squeeze(cv2.meanStdDev(image_S, img_binary_d), -1)
                            v_mean, v_std = np.squeeze(cv2.meanStdDev(image_V, img_binary_d), -1)

                            # h_std = cv2.std(image_H)  # 二阶颜色矩
                            # s_std = cv2.std(image_S)
                            # v_std = cv2.std(image_V)

                            h_skewness = cv2.mean(abs(image_H - h_mean) ** 3, img_binary_d)[0]  # 三阶颜色矩
                            s_skewness = cv2.mean(abs(image_S - s_mean) ** 3, img_binary_d)[0]
                            v_skewness = cv2.mean(abs(image_V - v_mean) ** 3, img_binary_d)[0]
                            h_thirdMoment = h_skewness ** (1. / 3)
                            s_thirdMoment = s_skewness ** (1. / 3)
                            v_thirdMoment = v_skewness ** (1. / 3)

                            features = np.append(
                                [aspectio, circularity, dConvexity, defects_num, equi_radius, smoothness,
                                 h_mean, s_mean, v_mean, h_std, s_std, v_std, h_skewness, s_skewness, v_skewness,
                                 humoments[0][0], humoments[1][0], humoments[2][0], humoments[3][0], humoments[4][0],
                                 humoments[5][0], humoments[6][0]], img_hog)

                            features = features.reshape((1,1318))
                            for j in range(1318):
                                max_j = float(a[j*2])
                                min_j = float(b[j*2])
                                features_j = float(features[0][j])
                                features[0][j] = (features_j - min_j)/(max_j - min_j)
                            # features = scalar.transform(features)
                            features = features.astype(np.float32)
                            _, y_predict = svm.predict(features)
                            y_predict = np.squeeze(y_predict)
                            print(time.time()-time_start)
                            # cv2.imencode('.png', img_binary_d)[1].tofile(dir[:-4]+ '.png')
                            # if not str(int(y_predict)) == dirname:
                            #     shutil.copy(dir, os.path.join(r'E:\细胞\hongxibao\error',dirname,image_id))
                            shutil.copy(dir, os.path.join(r'E:\细胞\hongxibao\test',str(int(y_predict)),image_id))

