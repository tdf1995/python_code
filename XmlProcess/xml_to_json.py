import xmltodict
import json
import os
import glob

root_path = r'E:\乌冬面\数据\已标注'
def xml_to_json(xml_path):
    xml_file = open(xml_path, 'r')
    xml_str = xml_file.read()
    json = xmltodict.parse(xml_str)
    return json
if __name__ == '__main__':
    xml_files = glob.glob(root_path + '/*.xml')
    for file in xml_files:
        json = xml_to_json(file)
        json_file = open(file[:-4]+'.json','w',encoding='UTF-8')
        json_file.write(str(json))
        json_file.close()
