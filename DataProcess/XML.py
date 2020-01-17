# 这个文件目标将xml点坐标转化为框坐标
from xml.dom import minidom
from os import walk
import os
import shutil
import codecs
pic_outpath = 'D:/guanxing/'
# xml_path = 'D:/xml/baixibao/'
# xml_path = 'D:/xml/hongxibao/'
# xml_path = 'D:/xml/jiejing/'
xml_path = 'D:/尿液低倍/xml/低倍管型/'

# pic_inpath = 'D:/GT_FCIS/baixibao/'
# pic_inpath = 'D:/GT_FCIS/hongxibao/'
# pic_inpath = 'D:/GT_FCIS/jiejing/'
pic_inpath = 'D:/尿液低倍/图片/低倍管型/'

def splitXml(dom):
    objs = dom.getElementsByTagName("object")
    root = dom.documentElement
    # print(root)
    for obj in objs:
        root.removeChild(obj)
    return dom, objs

def getFileName(root):
    filename = root.getElementsByTagName("filename")
    return filename[0].childNodes[0].nodeValue

def getSize(root):
    nWidth = root.getElementsByTagName("width")
    width = int(nWidth[0].childNodes[0].nodeValue)
    nHeight = root.getElementsByTagName("height")
    height = int(nHeight[0].childNodes[0].nodeValue)
    return width, height

def getObjCls(obj):
    nCls = obj.getElementsByTagName("name")
    Cls = str(nCls[0].childNodes[0].nodeValue)
    return Cls

def getBoundBox(obj, width, height):
    x = []
    y = []
    # width, height = getSize(obj)
    xbndBox = obj.getElementsByTagName("points_x")
    x = xbndBox[0].childNodes[0].nodeValue.split(sep = ',')
    x = [float(x) for x in x if x]

    Xmax = float(x[0])
    Xmin = float(x[0])
    for i in x:
        if Xmax < float(i):
            Xmax = float(i)
            if Xmax >= width:
                Xmax = width -1
        if Xmin > float(i):
            Xmin = float(i)
            if Xmin < 1:
                Xmin = 1
    # xmax = float(x[0])
    ybndBox = obj.getElementsByTagName("points_y")
    y = ybndBox[0].childNodes[0].nodeValue.split(sep=',')
    y = [float(y) for y in y if y]
    Ymax = float(y[0])
    Ymin = float(y[0])
    for i in y:
        if Ymax < float(i):
            Ymax = float(i)
            if Ymax >= height:
                Ymax = height - 1
        if Ymin > float(i):
            Ymin = float(i)
            if Ymin < 1:
                Ymin = 1
    return int(Xmin), int(Ymin), int(Xmax), int(Ymax)

def getObjCenter(obj):
    bndBox = obj.getElementsByTagName("bndbox")
    childs = bndBox[0].childNodes
    Xmin = int(childs[1].childNodes[0].nodeValue)
    Ymin = int(childs[3].childNodes[0].nodeValue)
    Xmax = int(childs[5].childNodes[0].nodeValue)
    Ymax = int(childs[7].childNodes[0].nodeValue)
    return int((Xmax + Xmin) / 2),int((Ymax + Ymin) / 2)

# def getSize(obj):
#     width = obj.getElementsByTagName("width")
#     height = obj.getElementsByTagName("height")
#     return width, height
TxtFile = 'D:/niaoyedibei.txt'

if __name__ == "__main__":
    try:
            f = open(TxtFile,'a')
    except IOError:
            print(' file open error:')
    file = []
    for (dirpath, dirnames, filenames) in walk(xml_path):
        file.extend(filenames)
        break
    for xmlFileName in file:
        if(xmlFileName [-4:] != ".xml"):# [-4:]文件名后4个字符
            continue
        dom = minidom.parse(xml_path + xmlFileName)# 读取.xml文件
        dom, objs = splitXml(dom)
        # imgFileName =  # 得到对应的图片名
        imgFileName = getFileName(dom)  # 得到对应的图片名

        # NewimgFileName = "bai"+getFileName(dom)#由于不同文件夹下的图片名存在同名，故重命名
        # NewimgFileName = "hong" + getFileName(dom)
        # NewimgFileName = "jie" + getFileName(dom)
        # NewimgFileName = "zhen" + getFileName(dom)

        NewimgFileName = "guanxing" + getFileName(dom)

        if os.path.exists(os.path.join(pic_inpath,imgFileName)):
            shutil.copyfile(os.path.join(pic_inpath, imgFileName),os.path.join(pic_outpath, NewimgFileName))  # 从数据集中复制xml文件对应的原图

        rectW, rectH = getSize(dom) # 从xml中得到图片尺寸
        for i in range(objs.length):
            # xmlName = xmlFileName[:-4] + "-" + str(i) + xmlFileName[-4:]    # [:-4]文件名前4个字符
            # print(xmlName)
            Class = getObjCls(objs[i])
            # Cls = Class[:3]
            # if Cls in {'1.1','1.2','1.3','1.4','1.5','1.6','1.7','1.8'}:
            #     Class = "hongxibao"
            # elif Cls in {'2.1','2.2','2.3'}:
            #     Class = "baixibao"
            # elif Cls in {'3.1','3.2','3.3','3.4','3.5','3.6','3.7','3.8'}:
            #     Class = "jiejing"
            # elif Cls in {'4.0'}:
            #     Class = "zhenjun"
            # elif Cls in {'5.0'}:
            #     Class = "xijun"
            # else:
            #     print("error")
            #     print(imgFileName)
            xmin, ymin, xmax, ymax = getBoundBox(objs[i], rectW, rectH)
            f.write(os.path.join(NewimgFileName[:-3]+'jpg')+' '+ Class + ' ' + str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' + str(ymax)+'\n')