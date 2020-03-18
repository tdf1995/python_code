import os
import glob
crop_path = r'\\192.168.1.251\ssd-研发部\项目工作目录\OCR项目\拍摄的数据集\字符区图片数据集\邦纳提供图片_字符区图_单字符矩形标注_crop_删除'
delete_path = r'\\192.168.1.251\ssd-研发部\项目工作目录\OCR项目\拍摄的数据集\字符区图片数据集\邦纳提供图片_字符区图_单字符矩形标注'

crop_files = glob.glob(crop_path+'/*/*.jpg')
for crop_file in crop_files:
    filename = os.path.basename(crop_file).split('_')[:-2]
    name= ''
    for i in range(len(filename)):
        name = name + filename[i]+'_'
    name= name[:-1]
    cls = os.path.basename(os.path.dirname(crop_file))
    delete_file_path = os.path.join(delete_path,cls,name+'.jpg')
    if os.path.exists(delete_file_path):
        os.remove(delete_file_path)
        os.remove(delete_file_path[:-4]+'.xml')
        print('1')