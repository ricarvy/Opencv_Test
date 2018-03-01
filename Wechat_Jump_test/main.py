# -*- coding: utf-8 -*-
# @Time    : 2018/1/18 18:47
# @Author  : Li Jiawei
# @FileName: main.py
# @Software: PyCharm
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def pull_screenshot():
    os.system('adb shell screencap -p /sdcard/autojump.png')
    os.system('adb pull /sdcard/autojump.png .')

img_rgb=cv2.imread('temp_data/0.png')
img_gray=cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)
template=cv2.imread('temp_data/temp_end.jpg',0)
w,h=template.shape[::-1]

res=cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold=0.8
loc=np.where(res>=threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb,pt,(pt[0]+w,pt[1]+h),(0,0,255),1)
plt.imshow(img_rgb)
plt.show()