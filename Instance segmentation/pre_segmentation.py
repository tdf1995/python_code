# 利用opencv算法预生成标注数据
import cv2
import os
import numpy as np
from skimage import io,morphology,exposure,color,filters

test_pic_path = r'D:\python code\Instance segmentation\for_pre_seg.jpg'
test_path = r'D:\python code\Instance segmentation/'

def nothing(x):
  pass
img = cv2.imdecode(np.fromfile(test_pic_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.namedWindow('Canny',0)
cv2.createTrackbar('minval','Canny',0,255,nothing)
cv2.createTrackbar('maxval','Canny',0,255,nothing)

while (1):

    key = cv2.waitKey(50) & 0xFF
    if key == ord('q'):
        break
    # 读取滑动条数值
    minval = cv2.getTrackbarPos('minval', 'Canny')
    maxval = cv2.getTrackbarPos('maxval', 'Canny')
    edges = cv2.Canny(img_gray, minval, maxval)

    # 拼接原图与边缘监测结果图
    img_2 = np.hstack((img_gray, edges))
    cv2.imshow('Canny', img_2)

cv2.destroyAllWindows()

_, img_bin = cv2.threshold(img_gray,240,255,cv2.THRESH_BINARY)
img_bin = ~img_bin
cv2.imencode('.jpg', img_bin)[1].tofile(test_path+'test_bin1.jpg')

contours, hierarch = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for i in range(len(contours)):
    area = cv2.contourArea(contours[i])
    if area < 300:
        cv2.drawContours(img_bin, [contours[i]], 0, 0, -1)
cv2.imencode('.jpg', img_bin)[1].tofile(test_path+'test_bin2.jpg')

kernel = np.ones((3, 3), np.uint8)
img_bin = cv2.morphologyEx(img_bin,cv2.MORPH_OPEN,kernel=kernel)
contours, hierarch = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for i in range(len(contours)):
    area = cv2.contourArea(contours[i])
    if area < 300:
        cv2.drawContours(img_bin, [contours[i]], 0, 0, -1)

cv2.imencode('.jpg', img_bin)[1].tofile(test_path+'test_bin3.jpg')



