import cv2
import numpy as np

def simple_show_mask(mask):
    _, mask_binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    return mask_binary

def mask_in_img(image, mask, draw_boundingRect=True,
                draw_minAreaRect=True,draw_contour=True):
    height = image.shape[0]
    width = image.shape[1]
    mask = cv2.resize(mask, (width,height),cv2.INTER_NEAREST)
    _, mask_binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    contours,hierarchy= cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours)>0:
        for i in range(len(contours)):
            if draw_boundingRect:
                x, y, w, h = cv2.boundingRect(contours[i])
                xmin, ymin, xmax, ymax = x, y, x + w, y + h
                cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(255,0,0),5)
            if draw_minAreaRect:
                rotated_rect = cv2.minAreaRect(contours[i])
                box = cv2.boxPoints(rotated_rect)
                box = np.int0(box)
                cv2.drawContours(image, [box], 0, (0, 255, 0), 5)
            if draw_contour:
                cv2.drawContours(image, contours, i, (0,0,255),5)
    return image

def crop_imgs(image, mask):
    height = image.shape[0]
    width = image.shape[1]
    mask = cv2.resize(mask, (width, height), cv2.INTER_NEAREST)
    _, mask_binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    num_contours = len(contours)
    rotated_img=[num_contours]
    for i in range(num_contours):
        rotated_rect = cv2.minAreaRect(contours[i])# 0为中心点坐标，1为宽高

        rect_max = int(rotated_rect[1][0]) if rotated_rect[1][0] > rotated_rect[1][1] else int(rotated_rect[1][1])
        rect_min = int(rotated_rect[1][1]) if rotated_rect[1][0] > rotated_rect[1][1] else int(rotated_rect[1][0])

        angle = rotated_rect[2] if abs(rotated_rect[2])<45 else (rotated_rect[2]+90)
        M = cv2.getRotationMatrix2D((int(rotated_rect[0][0]),int(rotated_rect[0][1])),angle,1)
        rotated_img[i] = cv2.warpAffine(image, M, (width,height))
        rotated_img[i] = cv2.getRectSubPix(rotated_img[i], (rect_max, rect_min),
                                        (int(rotated_rect[0][0]), int(rotated_rect[0][1])))
    return rotated_img
