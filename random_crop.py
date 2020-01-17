#随机裁剪图像（同比例）再resize到相同尺寸
import cv2
import os
import numpy as np
import random

crop_rate = 0.6

def random_crop(image, crop_shape, padding=None):
    img_h = image.shape[0]
    img_w = image.shape[1]
    img_d = image.shape[2]

    if padding:
        oshape_h = img_h + 2 * padding
        oshape_w = img_w + 2 * padding
        img_pad = np.zeros([oshape_h, oshape_w, img_d], np.uint8)
        img_pad[padding:padding + img_h, padding:padding + img_w, 0:img_d] = image

        nh = random.randint(0, oshape_h - crop_shape[0])
        nw = random.randint(0, oshape_w - crop_shape[1])
        image_crop = img_pad[nh:nh + crop_shape[0], nw:nw + crop_shape[1]]

        return image_crop
    else:
        print("WARNING!!! nothing to do!!!")
        return image

def preprocess(root_path, output_height, output_width):
   rootdir = os.listdir(root_path)
   for e in rootdir:
       subdir = os.path.join(root_path,e)   #   子文件及子文件夹路径
       if os.path.isfile(subdir):   #   如果是文件
            if os.path.splitext(subdir)[1] in {'.jpg','.bmp','.png'}:
                img = cv2.imdecode(np.fromfile(subdir,dtype=np.uint8),-1)
                seed = random.randint(output_height,int(output_height/crop_rate))
                seed_width =int(seed/output_height*output_width)
                img = cv2.resize(img, (seed_width, seed), interpolation=cv2.INTER_NEAREST)
                crop_shape = [output_height, output_width]
                img = random_crop(img, crop_shape, 1)
                if os.path.splitext(subdir)[1] == '.jpg':
                    cv2.imencode('.jpg', img)[1].tofile(subdir)
                if os.path.splitext(subdir)[1] == '.bmp':
                    cv2.imencode('.bmp', img)[1].tofile(subdir)
                if os.path.splitext(subdir)[1] == '.png':
                    cv2.imencode('.png', img)[1].tofile(subdir)
       elif os.path.isdir(subdir):  #   如果是路径
           preprocess(subdir, output_height, output_width)
