# 灰度模板匹配
import cv2
import numpy as np
from  matplotlib import pyplot as plt
import os

templatePath = r'C:\Users\tdf\Desktop/template.jpg'
root_path = r'D:\蒜蓉辣酱'
template = cv2.imdecode(np.fromfile(r'C:\Users\tdf\Desktop/template.jpg',dtype=np.uint8),-1)
template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
_, w, h = template.shape[::-1]


def MatchTemplate(root_path):
    rootdir = os.listdir(root_path)
    for e in rootdir:
        subdir = os.path.join(root_path, e)  # 子文件及子文件夹路径
        if os.path.isfile(subdir):  # 如果是文件
            if os.path.splitext(subdir)[1] in {'.jpg', '.bmp', '.png'}:
                img = cv2.imdecode(np.fromfile(subdir, dtype=np.uint8), -1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                method = eval('cv2.TM_CCORR')

                res = cv2.matchTemplate(img, template, method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
                if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                    top_left = min_loc
                else:
                    top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2.rectangle(img, top_left, bottom_right, 255, 2)
                img = img[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]

                if os.path.splitext(subdir)[1] == '.jpg':
                    cv2.imencode('.jpg', img)[1].tofile(subdir)
                if os.path.splitext(subdir)[1] == '.bmp':
                    cv2.imencode('.bmp', img)[1].tofile(subdir)
                if os.path.splitext(subdir)[1] == '.png':
                    cv2.imencode('.png', img)[1].tofile(subdir)
                # plt.subplot(121), plt.imshow(res, cmap='gray')
                # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
                # plt.subplot(122), plt.imshow(img, cmap='gray')
                # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
                # plt.suptitle('cv2.TM_CCORR')
                #
                # plt.show()

        elif os.path.isdir(subdir):  # 如果是路径
            MatchTemplate(subdir)
if __name__ == '__main__':
    MatchTemplate(root_path)