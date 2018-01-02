# -*- coding: utf-8 -*-
# @Time    : 2018/1/1 14:43
# @Author  : Li Jiawei
# @FileName: drawing_test.py
# @Software: PyCharm

import cv2
import numpy as np
img=np.zeros((1024,1024,3),np.uint8)

# ### draw a line
# line=cv2.line(img=img,pt1=(0,0),pt2=(511,511),color=(0,133,0),thickness=5)
# cv2.imshow('line',line)


# ### draw a rectangle
# rect=cv2.rectangle(img,pt1=(0,0),pt2=(511,511),color=(0,133,0),thickness=5)
# cv2.imshow('rectangle',rect)

# ### draw a circle
# circle=cv2.circle(img,center=(255,255),radius=190,color=(0,133,0),thickness=5)
# cv2.imshow('circle',circle)

# ### draw a ellipse
# ellipse=cv2.ellipse(img,center=(256,256),axes=(100,50),angle=90,startAngle=0,endAngle=360,color=255,thickness=5)
# cv2.imshow('ellipse',ellipse)

cv2.waitKey()
cv2.destroyAllWindows()