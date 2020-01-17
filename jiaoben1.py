import os
import shutil
root_path = r'E:\多目标跟踪\Market-1501-v15.09.15\VOCdevkit\VOC2007\已标注\Annotations'
target_path = r'E:\多目标跟踪\Market-1501-v15.09.15\VOCdevkit\VOC2007\JPEGImages'

pic_files = os.listdir(root_path)

for pic_file in pic_files:
    files = os.listdir(target_path)
    # inst_file = pic_file
    mask_file = pic_file[:-4]+'.jpg'
    if mask_file in files:
        # os.remove(os.path.join(root_path,inst_file))
        shutil.move(os.path.join(target_path,mask_file), os.path.join(root_path,mask_file))
    # if mask_file in files:
    #     shutil.move(os.path.join(target_path,mask_file), os.path.join(root_path,mask_file))