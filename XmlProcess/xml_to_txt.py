import os
import xml.etree.ElementTree as ET
import numpy as np
import sys

xml_path = r'D:\OCR\PixelLink\train_data\train_data_east/'

if __name__ == '__main__':
    files = os.listdir(xml_path)
    for e in files:
        file_path = os.path.join(xml_path, e)
        if os.path.splitext(file_path)[1] == '.xml':
            txt_file = open(xml_path+'gt_'+e[:-4]+'.txt',"w")
            tree = ET.parse(file_path)
            root = tree.getroot()
            boxes = []
            labels = []
            for obj in root.findall('object'):
                label = obj.find('name').text
                x = np.array(
                    [int(float(c)) for c in list(obj.find('polygen').find("points_x").text.strip().split(",")[0:4])])
                y = np.array(
                    [int(float(c)) for c in list(obj.find('polygen').find("points_y").text.strip().split(",")[0:4])])

                box = np.stack((x, y), axis=1)
                # boxes.append(box)
                # labels.append(label)
                txt_file.write(str(box[0,0])+','+str(box[0,1])+','+str(box[1,0])+','+str(box[1,1])+','+
                                str(box[2, 0]) + ',' +str(box[2,1])+','+str(box[3,0])+','+str(box[3,1])+','+label[2:]+'\n')
            txt_file.close()


