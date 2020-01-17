import xml.etree.ElementTree as ET
import os
import glob

xml_path = r'F:\models-master\cell\VOCdevkit\VOC2007\Annotations'
pic_path = r'F:\models-master\cell\VOCdevkit\VOC2007\JPEGImages/'

def xml_pic_Delete(xml_path, pic_path):
    xml_files = glob.glob(xml_path+'/*.xml')
    pic_files = glob.glob(pic_path+'/*.jpg')
    for xml_file in xml_files:
        # img = cv2.imdecode(np.fromfile(pic_file, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        # height, width, channel = img.shape
        tree = ET.parse(xml_file)
        root = tree.getroot()
        objs = root.findall('object')
        if objs is None or len(objs)==0:
            os.remove(xml_file)
            pic_p = os.path.join(pic_path,os.path.basename(xml_file)[:-4]+'.jpg')
            os.remove(pic_p)

if __name__ == '__main__':
    xml_pic_Delete(xml_path, pic_path)