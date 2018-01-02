# -*- coding: utf-8 -*-
# @Time    : 2017/12/31 14:06
# @Author  : Li Jiawei
# @FileName: video_test.py
# @Software: PyCharm

import cv2
import numpy as np

### catch frame from personal camera
# cap=cv2.VideoCapture(0)
# while(True):
#     ret,frame=cap.read()
#     gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     cv2.imshow('frame',gray)
#     if cv2.waitKey() == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

### catch frame from existed video
# cap=cv2.VideoCapture('data/test_video.mp4')
# while(cap.isOpened()):
#     ret,frame=cap.read()
#     cv2.imshow('frame',frame)
#     if cv2.waitKey()==ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

### using VideoWriter to write a frame set to disk
cap=cv2.VideoCapture(0)
fourcc=cv2.VideoWriter_fourcc(*'DIVX')
out=cv2.VideoWriter('data/test_video.mp4',fourcc,20.0,(640,480))

while(cap.isOpened()):
    ret,frame=cap.read()
    if ret == True:
        frame=cv2.flip(frame,0)
        out.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey()==ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()