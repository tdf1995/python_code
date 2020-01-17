# 这个程序根据ins，cls灰度图生成框的txt文件，结合matlab程序生成voc风格的xml文件
import cv2
import os
from os import walk
from matplotlib import pyplot as plt
import numpy as np

def getCenter(xmin, ymin, w, h):
    return int( xmin+ w / 2), int( ymin+ h/ 2)
def getCoordinate(x, y, w, h):
    if x == 0:
        xmin = x+1
    else:
        xmin = x
    if y == 0:
        ymin = y+1
    else:
        ymin = y
    if x+w <= 447:
        xmax = x + w
    else:
        xmax = 447
    if y+h <= 447:
        ymax = y + h
    else:
        ymax = 447
    return xmin, ymin, xmax, ymax

picPath = 'D:\gjpg1'
TxtFile = 'D:\Data.txt'

if __name__ == '__main__':

    try:
            f = open(TxtFile,'w')
    except IOError:
            print(' file open error:')

    for (dirpath, dirnames, filenames) in walk(picPath):
        for filename in filenames:# 文件名
            im_file = os.path.join(picPath, filename)# 绝对地址
            img = cv2.imread(im_file,cv2.IMREAD_UNCHANGED)
            # gray = cv2.imread(im_file, cv2.IMREAD_GRAYSCALE)
            # print(im_file)
            # cv2.imshow("contours", img)
            # f.write(filename+' ')
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # cv2.imshow("contou", img)
            # plt.imshow(img)
            # plt.show()
            ret, thresh = cv2.threshold(img, 1, 255, 0, cv2.THRESH_BINARY)
            # cv2.imshow("contou1", thresh)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            # print('1')
            a = 0
            for c in contours:
                # 边界框:

                # find bounding box coordinates
                # boundingRect()将轮廓转化成(x,y,w,h)的简单边框,cv2.rectangle()画出矩形[绿色(0, 255, 0)]
                x, y, w, h = cv2.boundingRect(c)
                # print(x, y, w, h)

                xc, yc = getCenter(x,y,w,h)
                xmin, ymin, xmax, ymax = getCoordinate(x, y, w, h)
                if (xmax-xmin)*(ymax-ymin)>10:
                    a = a + 1
                    if img[yc, xc]==0:
                        for i in (y,yc):
                            for j in(x,xc):
                                if img[i,j]:
                                    print(img[i,j])
                                    f.write(os.path.join(filename[:-3]+'jpg')+' '+str(img[i, j]) + ' ' + str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' + str(ymax)+'\n')

                    else:
                        print(img[yc, xc])
                        f.write(os.path.join(filename[:-3]+'jpg')+' '+str(img[yc, xc])+' '+  str(xmin)+ ' ' + str(ymin)+ ' ' + str(xmax) + ' ' + str(ymax)+'\n')
            if a == 0:
                os.remove(os.path.join('D:/jpg1/',filename[:-3]+'jpg'))
                os.remove(os.path.join('D:/gjpg1/',filename[:-3] +'png'))
    f.close()
                # print(xc, yc)
                #   img[yc, xc]+ + ' x' + y + ' ' + xm + ' ' + ym + '\n'
                # print(img.name)
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#     rect = cv2.minAreaRect(c)
#     # calculate coordinates of the minimum area rectangle
#     box = cv2.boxPoints(rect)
#     # normalize coordinates to integers
#     box = np.int0(box)
#     # draw contours
#     cv2.drawContours(img, [box], 0, (0, 0, 255), 3)
# cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
# cv2.imshow("contours", img)
#
# cv2.waitKey()
# cv2.destroyAllWindows()
# plt.hist(img.ravel(),256,[0,256])
# plt.show()