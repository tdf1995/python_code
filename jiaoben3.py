import cv2
import os
from os import walk
import numpy as np
import random
import glob
import re


def circle_crop(image, p=0.5):
    (height, width, _) = image.shape
    (center_y, center_x) = (height/2, width/2)
    L = min(height, width)
    image = image[int(center_y-L/2):int(center_y+L/2),int(center_x-L/2):int(center_x+L/2)]
    radius = int(L/2)
    mask = np.zeros((L,L),np.uint8)

    cv2.circle(mask, (radius, radius), int(radius*p),(255,255,255),-1)
    cv2.imshow('', mask)
    cv2.waitKey(0)
    image = cv2.bitwise_and(image, image ,mask = mask)
    cv2.imshow('1',image)
    cv2.waitKey(0)


if __name__ =='__main__':
    img = cv2.imread('20191029103627484_0.jpg')
    circle_crop(img,0.7)