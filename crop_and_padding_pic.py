# 这个程序目的是将木材原图剪裁背景并填充到固定尺寸
import cv2
import os
from os import walk
import numpy as np
from math import ceil

root_path = r'F:\models-master\cell\分割\datasets\pascal_voc_seg\VOCdevkit\VOC2012/'


max_w = 0
min_w = 10000
total_w = 0
max_h = 0
min_h = 10000
total_h = 0
max_ratio = 0
min_ratio = 10
total_ratio = 0


def Get_w_and_h(img):
    global max_w
    global max_h
    global min_w
    global min_h
    global max_ratio
    global min_ratio
    global total_h
    global total_w
    global total_ratio
    # Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(Gray, 30, 255, 0, cv2.THRESH_BINARY)
    # thresh, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    #
    # for contour in contours:
    #     area = cv2.contourArea(contour)
    #     if area < 1000:
    #         continue
    #     x, y, w, h = cv2.boundingRect(contour)
    #     if w > max_w:
    #         max_w = w
    #     if h > max_h:
    #         max_h = h
    h, w, _ = img.shape
    total_w = total_w + w
    total_h = total_h +h
    ratio = w/h
    total_ratio =total_ratio+ratio
    # f.write("高："+str(h)+' '+"宽："+str(w)+' '+"比："+str(ratio)+'\n')

    if w > max_w:
        max_w =w
    if w < min_w:
        min_w = w
    if h > max_h:
        max_h = h
    if h < min_h:
        min_h = h
    if w/h >max_ratio:
        max_ratio = w/h
    if w / h < min_ratio:
        min_ratio = w / h
    print(max_w,max_h)

# def Crop_and_Padding(img,direpath, filename):
#     # global max_w
#     # global max_h
#     Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # dst = cv2.adaptiveThreshold(Gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
#     ret, dst = cv2.threshold(Gray,35,255,cv2.THRESH_BINARY)
#
#     thresh, contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # cv2.drawContours(img,contours,-1,(0,0,255),3)
#     # cv2.imshow("thresh", img)
#     # cv2.waitKey(0)
#     # print(filename)
#
#     for contour in contours:
#
#         area = cv2.contourArea(contour)
#         print(area)
#         if area > 5000:
#             x, y, w, h = cv2.boundingRect(contour)
#             cropImg = img[y:y+h, x:x+w]
#             outpath = os.path.join(output_pic_path, dirname, filename)
#             cv2.imwrite(outpath, cropImg)
#             break



    # paddingImg= cv2.copyMakeBorder(cropImg,int((max_h-h)/2),int((max_h-h)/2),int((max_w-w)/2),int((max_w-w)/2),cv2.BORDER_CONSTANT,value=[0,0,0])

    # cv2.imwrite(output_pic_path+filename,paddingImg)
    #



if __name__ == '__main__':
    # f = open(txt,'w')
    for (dirpath, dirnames,_) in walk(root_path):
        for dirname in dirnames:
            direpath = os.path.join(root_path, dirname)
            for(_,_,filenames)in walk(direpath):
                for filename in filenames:
                    img = cv2.imdecode(np.fromfile(os.path.join(direpath, filename), dtype=np.uint8), -1)
                    Get_w_and_h(img)
                    # h, w, _ = img.shape
                    # Padding_img = cv2.copyMakeBorder(img, int((310 - h) / 2), ceil((310 - h) / 2),
                    #                      int((310 - w) / 2), ceil((310 - w) / 2), cv2.BORDER_CONSTANT,
                    #                      value=[0, 0, 0])
                    # cv2.imencode('.jpg', Padding_img)[1].tofile(os.path.join(r'F:\models-master\cell\分类\test_padding',dirname, filename))
                    # Crop_and_Padding(img, direpath,filename)


        # for filename in filenames:
        #     img = cv2.imread(Origin_pic_path + filename, cv2.IMREAD_UNCHANGED)
        #             Crop_and_Padding(img, filename)