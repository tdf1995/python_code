#根据xml生成分割类别png
import os
from xml.dom import minidom
import cv2
import numpy as np

def get_bounding_point(dom):
    points_b_list=[]
    objs = dom.getElementsByTagName("object")
    label_list=[]
    for obj in objs:
        points_b=[]
        labelNode=obj.getElementsByTagName("name")
        labelStr=labelNode[0].childNodes[0].nodeValue
        label_str_list=labelStr.split('.')
        # label=int(label_str_list[0])
        # label_list.append(label)
        polygen=obj.getElementsByTagName("polygen")
        pxNode=polygen[0].getElementsByTagName("points_x")
        pyNode=polygen[0].getElementsByTagName("points_y")
        px_str=pxNode[0].childNodes[0].nodeValue
        py_str =pyNode[0].childNodes[0].nodeValue

        px_=px_str.split(',')
        py_=py_str.split(',')

        for i in range(len(px_)):
            if px_[i]!='':
                points_b.append((int(float(px_[i])),int(float(py_[i]))))

        points_b_list.append(points_b)

    return points_b_list,label_list

def gen_cls_img():

    folder_path = r'E:\菜品\分割\训练标注数据\第三批'

    img_files = os.listdir(os.path.join(folder_path))
    for file in img_files:
        if file.endswith('jpg'):
            xml_file_path = os.path.join(folder_path, file[:-3] + 'xml')
            if not os.path.exists(xml_file_path):
                print(xml_file_path, 'not found!')
                imgPath = os.path.join(folder_path, file)
                imgOri = cv2.imdecode(np.fromfile(imgPath, dtype=np.uint8), cv2.IMREAD_COLOR)
                h, w, d = imgOri.shape
                groundTruth = np.zeros((h, w, 3), dtype=np.uint8)
                groundTruth = cv2.cvtColor(groundTruth, cv2.COLOR_RGB2GRAY)
                pathMask = os.path.join(folder_path, file[:-4] + '.png')
                # cv2.imwrite(pathMask, groundTruth)
                cv2.imencode('.png', groundTruth)[1].tofile(pathMask)
                continue

            pathMask = os.path.join(folder_path, file[:-4] + '.png')
            if not os.path.exists(pathMask):

                dom = minidom.parse(xml_file_path)
                pt_bounding, labels = get_bounding_point(dom)
                # img_filename = get_img_filename(dom)
                imgPath = os.path.join(folder_path, file)
                size = dom.getElementsByTagName("size")
                wNode = size[0].getElementsByTagName("width")
                hNode = size[0].getElementsByTagName("height")
                w = int(wNode[0].childNodes[0].nodeValue)
                h = int(hNode[0].childNodes[0].nodeValue)
                # imgOri = cv2.imread(imgPath)
                imgOri = cv2.imdecode(np.fromfile(imgPath, dtype=np.uint8), cv2.IMREAD_COLOR)
                # h, w, d = imgOri.shape
                # img = np.zeros((h, w, d), dtype=np.uint8)
                # groundTruth = np.zeros((h, w, 3), dtype=np.uint16)
                groundTruth = np.zeros((h, w, 3), dtype=np.uint8)

                for n in range(len(pt_bounding)):
                    # label=labels[n]
                    label = 1
                    if label >= 1 and label <= 30:
                        cv2.fillPoly(groundTruth, np.array([pt_bounding[n]], np.int32), (label, label, label),
                                     cv2.LINE_8)
                        print(label)
                    else:
                        print(xml_file_path, label)

                groundTruth = cv2.cvtColor(groundTruth, cv2.COLOR_RGB2GRAY)
                pathMask = os.path.join(folder_path, file[:-4] + '.png')
                # cv2.imwrite(pathMask, groundTruth)
                cv2.imencode('.png', groundTruth)[1].tofile(pathMask)

if __name__ =='__main__':
    gen_cls_img()