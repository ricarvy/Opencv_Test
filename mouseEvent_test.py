# -*- coding: utf-8 -*-
# @Time    : 2018/1/1 14:58
# @Author  : Li Jiawei
# @FileName: mouseEvent_test.py
# @Software: PyCharm

import cv2
import numpy as np
events=[i for i in dir(cv2) if 'EVENT' in i]
print(events)
img=np.zeros((512,512,3),np.uint8)

def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print('draw!')
        cv2.circle(img=img,center=(x,y),radius=100,color=(255,0,0),thickness=5)

cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)
while(1):
    cv2.imshow('image',img)
    if cv2.waitKey()== ord('q'):
        break
cv2.destroyAllWindows()