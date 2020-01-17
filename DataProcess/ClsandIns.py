#这个文件旨在于根据xml文件得到两张灰度图：
#根据类别，一个类别一个灰度值
#根据实例，一个细胞一个灰度值
import cv2
import  os
from xml.dom import minidom
from os import walk
import shutil
import XML
import numpy as np
# import matplotlib.pyplot as plt


pic_path = 'D:/1'   # opencv imwrite不能写中文路径
xml_path = 'D:/尿液低倍/xml/低倍粘液丝/'#xml文件位置
img_path = 'D:/jpg/'#原图位置，opencv imread 不能读中文路径
def getBox(obj):
    xbox = []
    ybox = []
    xbndBox = obj.getElementsByTagName("points_x")
    xbox = xbndBox[0].childNodes[0].nodeValue.split(sep=',')
    ybndBox = obj.getElementsByTagName("points_y")
    ybox = ybndBox[0].childNodes[0].nodeValue.split(sep=',')
    return xbox, ybox
if __name__ == "__main__":
    file = []
    num_1 = 0
    num_2 = 0
    num_3 = 0
    num_4 = 0
    # num_5 = 0
    # num_6 = 0
    for (dirpath, dirnames, filenames) in walk(xml_path):
        file.extend(filenames)
        break
    for xmlFileName in file:# 每一个xml文件
        if(xmlFileName [-4:] != ".xml"):# [-4:]文件名后4个字符
            continue
        dom = minidom.parse(xml_path + xmlFileName)
        # width, height = XML.getSize(dom)
        imgFileName = XML.getFileName(dom)#得到对应的原图
        jpg_path = os.path.join(img_path, imgFileName)
        img = cv2.imread(jpg_path, cv2.IMREAD_UNCHANGED)
        # width = img.width
        # height = img.height
        [width, height, channel] = img.shape
        Cls_Pic = np.zeros([width, height], dtype=np.uint8)# 创造一个空白的用来表达类别信息的图像
        Ins_Pic = np.zeros([width, height], dtype=np.uint16)# 创造一个空白的用来表达实例信息的图像
        # cv2.imshow("1",Cls_Pic)
        # cv2.waitKey(0)

        # print(imgFileName)
        dom, objs = XML.splitXml(dom)
        Cell_num = 0
        for obj in objs:# 每个细胞
            Cell_num = Cell_num + 1
            box=[]
            Cls = XML.getObjCls(obj)
            # print(Cls)
            Gray = Cls[:1]
            if not Gray in {'1','2','3','4'}:
                print(Cls)
                print(xml_path,imgFileName)
                continue
            if Gray == '1':
                num_1 = num_1 + 1
            elif Gray == '2':
                num_2 = num_2 + 1
            elif Gray == '3':
                num_3 = num_3 + 1
            elif Gray == '4':
                num_4 = num_4 + 1
            # elif Gray == '5':
            #     num_5 = num_5 + 1
            # elif Gray == '6':
            #     num_6 = num_6 + 1
            #     print(imgFileName)
            x, y = getBox(obj)
            px = []
            py = []
            point_num = len(x)-1
            ps = np.zeros([point_num, 2], dtype=np.int32)
            for i in range(point_num):

                if not x[i]:
                    print('error')
                    continue
                px = np.append(px, x[i])
                py = np.append(py, y[i])
                ps[i] = (int(float(px[i])), int(float(py[i])))
            # print(ps)
            cv2.polylines(Cls_Pic,[ps],1,0,1)#img:图像,顶点集，是否闭合，颜色，线宽度
            cv2.fillPoly(Cls_Pic, [ps], int(Gray))# 给每个类别的细胞一个对应的灰度

            cv2.polylines(Ins_Pic, [ps], 1, 0, 1)  # img:图像,顶点集，是否闭合，颜色，线宽度
            cv2.fillPoly(Ins_Pic, [ps], Cell_num)  # 给每个细胞一个对应的灰度
            # cv2.imshow("1",Cls_Pic)
            # cv2.waitKey(0)
        Cls_Pic_name = imgFileName[:-4] + "_cls.png"
        Ins_Pic_name = imgFileName[:-4] + "_ins.png"
        dir = os.path.join(pic_path,Cls_Pic_name)
        # cv2.imshow('1',Cls_Pic)
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join(pic_path,imgFileName),img)
        cv2.imwrite(os.path.join(pic_path,Cls_Pic_name),Cls_Pic,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(os.path.join(pic_path, Ins_Pic_name), Ins_Pic,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        # plt.savefig(Cls_Pic)
    print(num_1)
    print(num_2)
    print(num_3)
    print(num_4)
    # print(num_5)
    # print(num_6)

