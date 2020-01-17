#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/18 10:46
# @Author  : He Hangjiang
# @Site    :
# @File    : 根据目标图片id筛选图片.py
# @Software: PyCharm

import json
import os

nameStr = []

with open("COCO_train.json", "r+") as f:
    data = json.load(f)
    print("read ready")

for i in data:
    imgName = "000000" + str(i["filename"]) + ".jpg"
    nameStr.append(imgName)

nameStr = set(nameStr)
print(nameStr)
print(len(nameStr))

path = "D:/dataset/coco/train2017/train2017/"#118287

for file in os.listdir(path):
    if (file not in nameStr):
        if os.path.exists(path + file):
            os.remove(path + file)
        else:
            continue