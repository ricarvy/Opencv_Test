# -*- coding: utf-8 -*-
# @Time    : 2017/12/31 14:06
# @Author  : Li Jiawei
# @FileName: video_test.py
# @Software: PyCharm

import cv2
import numpy as np

cap=cv2.VideoCapture(0)

while(True):
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',gray)
    if cv2.waitKey() == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()