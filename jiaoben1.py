import os
import shutil
root_path = r'\\192.168.1.251\ssd-研发部\项目工作目录\OCR项目\拍摄的数据集\检测ROI图片数据集\原图'
target_path = r'\\192.168.1.251\ssd-研发部\项目工作目录\OCR项目\拍摄的数据集\检测ROI图片数据集\png'

pic_files = os.listdir(root_path)
files = os.listdir(target_path)
for pic_file in pic_files:
    # inst_file = pic_file
    mask_file = pic_file[:-4]+'.png'
    if mask_file not in files:
        os.remove(os.path.join(root_path,pic_file))
        # shutil.move(os.path.join(root_path,pic_file), os.path.join(target_path,pic_file))
    # if mask_file in files:
    #     shutil.move(os.path.join(target_path,mask_file), os.path.join(root_path,mask_file))