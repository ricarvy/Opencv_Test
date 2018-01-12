# -*- coding: utf-8 -*-
# @Time    : 2018/1/4 19:25
# @Author  : Li Jiawei
# @FileName: Harris_Corner_Detection.py
# @Software: PyCharm

import cv2
import numpy as np
import matplotlib as plt

def rgb_bgr(img):
    (r,g,b)=cv2.split(img)
    return cv2.merge((b,g,r))

filename='ali.jpg'
img=cv2.imread(filename=filename)
img=rgb_bgr(img)

# # find Harris corners
# # gray=np.float32(gray)
# dst=cv2.cornerHarris(gray,2,3,0.04)
# #
# dst=cv2.dilate(dst,None)
#
# gray[dst>0.01*dst.max()]=[0,0,225]
cv2.imshow('dst',img)

cv2.waitKey()
cv2.destroyAllWindows()