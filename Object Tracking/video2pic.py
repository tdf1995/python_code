import cv2
import os
import glob

video_path = r'E:\多目标跟踪\1.mp4'

vc = cv2.VideoCapture(video_path)
c=0
rval=vc.isOpened()
#timeF = 1  #视频帧计数间隔频率
while rval:   #循环读取视频帧
    c = c + 1
    rval, frame = vc.read()
#    if(c%timeF == 0): #每隔timeF帧进行存储操作
#        cv2.imwrite('smallVideo/smallVideo'+str(c) + '.jpg', frame) #存储为图像
    if rval:
        #存储为图像
        cv2.imencode('.jpg', frame)[1].tofile(r'E:\多目标跟踪\pic\1/'+str(c).zfill(8) + '.jpg')
        cv2.waitKey(10)
    else:
        break
vc.release()
