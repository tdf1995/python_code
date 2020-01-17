# 这个程序根据txt文件给对应的图像画目标框
import re
import os
from os import walk
import cv2

txt_path = 'D:/gaobeitxt/'
pic_path = 'D:/gaobeibiaozhun/image/'
out_path = 'D:/2/'

if __name__ == "__main__":
    for dirpath, dirnames, txtnames in walk(txt_path):
        for txtname in txtnames:
            txtfile = open(txt_path+txtname,'r')
            pic_name = txtname[5:]
            pic = cv2.imread(pic_path+pic_name[:-4]+'.jpg',cv2.IMREAD_UNCHANGED)
            lines = txtfile.readlines()
            for line in lines:
                a = re.split('[ \n]', line)
                if len(a) == 4:
                    cell_cls = a[0]
                    cell_num = a[2]
                elif  len(a) == 9:
                    error = a.pop(-1)
                    for i in range(int(len(a)/8)):
                        xmin = int(a.pop(0))
                        ymin = int(a.pop(0))
                        ymax = int(a.pop(-1))
                        xmax = int(a.pop(-1))

                        if cell_cls == '1':
                            cv2.rectangle(pic, (xmin, ymin), (xmax, ymax), (0,0,255),1,16)
                            cv2.putText(pic, 'H',(xmin+3, ymin+10),cv2.FONT_ITALIC,0.4, (0,0,255), 1)
                        elif cell_cls == '2':
                            cv2.rectangle(pic, (xmin, ymin), (xmax, ymax), (0,255,0),1,16)
                            cv2.putText(pic, 'B', (xmin+3,ymin+10), cv2.FONT_ITALIC, 0.4, (0,255,0), 1)
                        elif cell_cls == '3':
                            cv2.rectangle(pic, (xmin, ymin), (xmax, ymax), (255,128,0),1,16)
                            cv2.putText(pic, 'J', (xmin+3, ymin+10), cv2.FONT_ITALIC, 0.4, (255,128,0), 1)
                        elif cell_cls == '4':
                            cv2.rectangle(pic, (xmin, ymin), (xmax, ymax), (255,0,255),1,16)
                            cv2.putText(pic, 'Z', (xmin+3, ymin+10), cv2.FONT_ITALIC, 0.4, (255,0,255), 1)
                        elif cell_cls == '5':
                            cv2.rectangle(pic, (xmin, ymin), (xmax, ymax), (0,255,255),1,16)
                            cv2.putText(pic, 'X', (xmin+3, ymin+10), cv2.FONT_ITALIC, 0.4, (0,255,255), 1)
                elif len(a) == 8:
                    for i in range(int(len(a) / 8)):
                        xmin = int(a.pop(0))
                        ymin = int(a.pop(0))
                        ymax = int(a.pop(-1))
                        xmax = int(a.pop(-1))

                        if cell_cls == '1':
                            cv2.rectangle(pic, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1,16)
                            cv2.putText(pic, 'H', (xmin+3, ymin+10), cv2.FONT_ITALIC, 0.4, (0, 0, 255), 1)
                        elif cell_cls == '2':
                            cv2.rectangle(pic, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1,16)
                            cv2.putText(pic, 'B', (xmin+3, ymin+10), cv2.FONT_ITALIC, 0.4, (0, 255, 0), 1)
                        elif cell_cls == '3':
                            cv2.rectangle(pic, (xmin, ymin), (xmax, ymax), (255, 128, 0), 1,16)
                            cv2.putText(pic, 'J', (xmin+3, ymin+10), cv2.FONT_ITALIC, 0.4, (255, 128, 0), 1)
                        elif cell_cls == '4':
                            cv2.rectangle(pic, (xmin, ymin), (xmax, ymax), (255, 0, 255), 1,16)
                            cv2.putText(pic, 'Z', (xmin+3, ymin+10), cv2.FONT_ITALIC, 0.4, (255, 0, 255), 1)
                        elif cell_cls == '5':
                            cv2.rectangle(pic, (xmin, ymin), (xmax, ymax), (0, 255, 255), 1,16)
                            cv2.putText(pic, 'X', (xmin+3, ymin+10), cv2.FONT_ITALIC, 0.4, (0, 255, 255), 1)
            cv2.imwrite(out_path+pic_name[:-4]+'.jpg', pic)
