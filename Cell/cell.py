# 这个文件通过cls,ins图像统计一些细胞的特征：细胞面积、细胞周长、似圆度、rotated box长宽比，饱和度、（灰度共生矩阵，信息复杂度）
import cv2
import os
import numpy as np
import math

Ins_Path = 'D:\GT_FCIS' #存放ins灰度图的路径
# Cls_Path = 'D:\Cell\cls_pic' #存放cls灰度图的路径

gray_level = 256

def MaxPixel(img, width, height):
    max_pixel = img[0,0]
    for i in range(width):
        for j in range(height):
            if img[i, j] >= max_pixel:
                max_pixel = img[i, j]
    return max_pixel
def getGlcm(input,d_x,d_y):
    srcdata=input.copy()
    ret=[[0.0 for i in range(gray_level)] for j in range(gray_level)]
    (height,width) = input.shape

    max_gray_level=gray_level

    #若灰度级数大于gray_level，则将图像的灰度级缩小至gray_level，减小灰度共生矩阵的大小
    # if max_gray_level > gray_level:
    #     for j in range(height):
    #         for i in range(width):
    #             srcdata[j][i] = srcdata[j][i]*gray_level / max_gray_level

    for j in range(height - d_y - 1):
        for i in range(width - d_x -1):
            # if srcdata[j][i] == 0:
            #     continue

            rows = srcdata[j][i]
            cols = srcdata[j + d_y][i+d_x]
            ret[rows][cols]+=1.0

    for i in range(gray_level):
        for j in range(gray_level):
            ret[i][j]/=float(height*width)

    return ret

def feature_computer(p):
    Con=0.0
    Eng=0.0
    Asm=0.0
    Idm=0.0
    for i in range(gray_level):
        for j in range(gray_level):
            Con+=(i-j)*(i-j)*p[i][j]
            Con = round(Con, 3)
            Asm+=p[i][j]*p[i][j]
            Asm = round(Asm, 4)
            Idm+=p[i][j]/(1+(i-j)*(i-j))
            Idm = round(Idm, 3)
            if p[i][j]>0.0:
                Eng+=p[i][j]*math.log(p[i][j])
                Eng = round(Eng, 3)
    return Asm,Con,-Eng,Idm

TxtFile = 'D:/Cell.txt'

if __name__ == '__main__':
    try:
            f = open(TxtFile,'w')# 'w'擦除，'a'之后
    except IOError:
            print(' file open error:')
    for root, dirs, InsFiles in os.walk(Ins_Path):
        for dir in dirs:
            Sub_Path = os.path.join(Ins_Path, dir)
            print(Sub_Path)
            for subroot, subdirs, SubInsFiles in os.walk(Sub_Path):
                for InsFile in SubInsFiles: # 每一张ins图
                    # print(InsFile[-7:])
                    if not InsFile[-7:] == 'ins.png':
                        continue
                    ins_file = cv2.imread(Sub_Path + "/" + str(InsFile),cv2.IMREAD_UNCHANGED)
                    pic_name = str(InsFile)[:-8]
                    # print(InsFile)
                    cls_file = cv2.imread(os.path.join(Sub_Path, pic_name + '_cls.png'),cv2.IMREAD_UNCHANGED) #对应的cls图，用来得到细胞的类别
                    pic_file = cv2.imread(os.path.join(Sub_Path, pic_name + '.jpg'),cv2.IMREAD_UNCHANGED) # 原图
                    pic_gray = cv2.cvtColor(pic_file, cv2.COLOR_BGR2GRAY)
                    print(os.path.join(Sub_Path, pic_name + '.jpg'))
                    width, height = ins_file.shape
                    # cv2.imshow("0", ins_file)
                    max_pixel = MaxPixel(ins_file, width, height)
                    if max_pixel == 0:
                        continue
                    # print(max_pixel)
                    # print(str(InsFile))
                    for i in range(1, max_pixel+1):
                        i = max_pixel - i
                        ret, thresh = cv2.threshold(ins_file, i, 65535, 0, cv2.THRESH_BINARY)

                        thresh_INV = cv2.bitwise_not(thresh)# 已读取的轮廓从图中抠除，此为掩膜
                        ins_file = cv2.bitwise_and(ins_file, thresh_INV)# 从图中抠除

                        ins_cell = thresh.astype(np.uint8)
                        pic_cell = cv2.bitwise_and(pic_gray, ins_cell)# 从原图中抠出细胞实例

                        # cv2.imshow("2", pic_cell)
                        # cv2.waitKey()
                        # cv2.destroyAllWindows()
                        # cv2.imshow("1", thresh)
                        image, contours, hierarchy = cv2.findContours(ins_cell, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        if len(contours) == 0:
                            print("Sorry No contour Found.")
                        else:
                            cnt = contours[0]
                            x, y, W, H = cv2.boundingRect(contours[0])
                            roiImg = pic_cell[y:y+H,x:x+W]
                            # cv2.imshow("1", roiImg)
                            # cv2.waitKey()
                            # cv2.destroyAllWindows()
                            if H > W:
                                temp = W
                                W = H
                                H = temp

                            M = cv2.moments(cnt)
                            # print(str(InsFile))
                            area = cv2.contourArea(cnt)# 计算轮廓面积
                            if M['m00'] == 0:
                                continue
                            cx = int(M['m10']/M['m00'])
                            cy = int(M['m01']/M['m00']) # 轮廓质心坐标
                            Cls = cls_file[cy, cx] #从cls图像得到细胞轮廓的类别
                            Class = []
                            if 1 <= Cls <= 8:
                                Class = 'hongxibao'
                            elif Cls == 9:
                                Class = 'baixibao'
                            elif Cls == 10:
                                Class = 'nongqiu'
                            elif 11 <= Cls <= 17:
                                Class = 'jiejing'
                            elif Cls == 18:
                                Class = 'luoanquanjiejing'
                            elif Cls == 19:
                                Class = 'zhenjun'
                            elif Cls == 20:
                                Class = 'xijun'
                            else:
                                print('error')
                                continue
                            # print(Class)

                            perimeter = round(cv2.arcLength(cnt,True), 3) #计算轮廓的周长
                            roundness = round((4 * math.pi * area)  / (perimeter**2), 3) #计算轮廓似圆度
                            rect = cv2.minAreaRect(cnt)
                            box = cv2.boxPoints(rect)
                            box = np.int0(box)
                            d = ((box[0][0]-box[1][0])**2 + (box[0][1]-box[1][1])**2)**0.5
                            h = ((box[1][0]-box[2][0])**2 + (box[1][1]-box[2][1])**2)**0.5
                            if h > d:
                                temp = d
                                d = h
                                h = temp
                            AspectRatio = round((d / h), 3) # 计算长宽比
                            saturation = round((area / (d*h)), 3)   #   计算饱和度
                            glcm_0 = getGlcm(roiImg, 1, 0)
                            glcm_1=getGlcm(roiImg, 0,1)
                            glcm_2=getGlcm(roiImg, 1,1)
                            glcm_3=getGlcm(roiImg, -1,1)

                            asm0, con0, eng0, idm0 = feature_computer(glcm_0)
                            asm1, con1, eng1, idm1 = feature_computer(glcm_1)
                            asm2, con2, eng2, idm2 = feature_computer(glcm_2)
                            asm3, con3, eng3, idm3 = feature_computer(glcm_3)
                            # cv2.imshow("2",ins_cell)
                            f.write(Class+" 面积："+str(area)+" 周长："+str(perimeter)+" 似圆度："+str(roundness)+" 长宽比："+str(AspectRatio)+" 饱和度："+str(saturation)+" 长："+str(W)+" 宽："+str(H)
                                    +" GLCM0:"+str(asm0)+" "+str(con0)+" "+str(eng0)+" "+str(idm0)
                                    +" GLCM1:"+str(asm1)+" "+str(con1)+" "+str(eng1)+" "+str(idm1)
                                    +" GLCM2:"+str(asm2)+" "+str(con2)+" "+str(eng2)+" "+str(idm2)
                                    +" GLCM3:"+str(asm3)+" "+str(con3)+" "+str(eng3)+" "+str(idm3)+"\n")
                            # print(area,perimeter,roundness,AspectRatio,saturation)






    f.close()