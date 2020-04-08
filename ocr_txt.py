import os
import glob
import cv2
import numpy as np
import os
import glob
import xml.etree.ElementTree as ET
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import re

pic_path = r'E:\ocr\shengcehng'
pics = glob.glob(pic_path+'\*.jpg')
f = open(r'E:\ocr\shengcehng\50.txt','w')
for pic in pics:
    img = cv2.imread(pic,cv2.IMREAD_GRAYSCALE)
    _, img_bin = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    # cv2.imshow('1',img_bin)
    # cv2.waitKey(0)
    ziti = os.path.basename(pic)[:-4]
    xml_path = pic[:-4]+'.xml'
    tree = ET.parse(xml_path)
    objs = tree.findall('object')

    txt_name = '50_' + os.path.basename(pic)[:-4]+'.txt'

    f.write(ziti+'*')
    for i, obj in enumerate(objs):
        xmin = int(obj.find('bndbox').find('xmin').text)
        ymin = int(obj.find('bndbox').find('ymin').text)
        xmax = int(obj.find('bndbox').find('xmax').text)
        ymax = int(obj.find('bndbox').find('ymax').text)
        name = obj.find('name').text
        crop_img = img_bin[ymin-5:ymax+5, xmin-5:xmax+5]
        h = crop_img.shape[0]
        w = crop_img.shape[1]
        # cv2.imshow('2', crop_img)
        # cv2.waitKey(0)
        offset_ymin = 0
        offset_xmin = 0
        offset_ymax = 0
        offset_xmax = 0
        for i in range(h):
            if (crop_img[i,:]==255).all():
                continue
            else:
                offset_xmin = i
                break
        for j in range(w):
            if (crop_img[:,j]==255).all():
                continue
            else:
                offset_ymin = j
                break
        for k in range(h):
            if (crop_img[h-k-1,:]==255).all():
                continue
            else:
                offset_xmax = k
                break
        for l in range(w):
            if (crop_img[:,w-l-1]==255).all():
                continue
            else:
                offset_ymax = l
                break
        f.write(name+'*'+str(offset_ymin-6)+'*'+str(offset_xmin-6)+'*'+str(offset_ymax-4)+'*'+str(offset_xmax-4)+'*')
    f.write('\n')
f.close()

    #
    # f = open(txt_path, 'w')
    # f.write('num:(1,5):(-3,0)'+'\n')
    # f.write('alphabet:(1,5):(-3,0)' + '\n')
    # f.write('char:(2,3):(-3,0)' + '\n')
    # f.close()
