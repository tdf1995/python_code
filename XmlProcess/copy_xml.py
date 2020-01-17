import os
import glob
import xml.etree.ElementTree as ET

xml_template = r'C:\Users\tdf\Desktop\20190927151455070.xml'
pic_path = r'E:\蒸浏记\分割\增强\1'

pics_path = glob.glob(pic_path+'\*.jpg')
for pic_path in pics_path:
    pic_name = os.path.basename(pic_path)
    tree = ET.parse(xml_template)
    root = tree.getroot()
    root.find('filename').text = pic_name
    tree.write(pic_path[:-4]+'.xml')