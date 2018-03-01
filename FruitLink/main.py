# -*- coding: utf-8 -*-
# @Time    : 2018/1/19 18:10
# @Author  : Li Jiawei
# @FileName: main.py
# @Software: PyCharm

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import time

STAGE=0

def get_screenshot(id):
    os.system('adb shell screencap -p /sdcard/%s.png' % str(id))
    os.system('adb pull /sdcard/%s.png .' % str(id))

def tab_screen(x,y):
    os.system('adb shell input tap %.2f %.2f'%(x,y))

def templateMatching(img,template,threshold):
    w,h = template.shape[::-1]
    res=cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
    min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(res)
    center1_loc = (max_loc1[0] + w//2, max_loc1[1] + h//2)
    return center1_loc

while(True):
    get_screenshot(0)
    screen = cv2.imread('0.png')
    plt.imshow(screen)
    plt.show()
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    if STAGE == 0:
        mode_template = cv2.imread('templates/classic_mode.png', 0)
        w, h = mode_template.shape[::-1]
        center = templateMatching(screen_gray, mode_template, 0.8)
        print(center[0] + w // 2, center[1] + h // 2)
        tab_screen(center[0] + w // 2, center[1] + h // 2)
        STAGE+=1
    if STAGE == 1:
        mode_template = cv2.imread('templates/single_mode.png', 0)
        w, h = mode_template.shape[::-1]
        center = templateMatching(screen_gray, mode_template, 0.8)
        print(center[0] + w // 2, center[1] + h // 2)
        tab_screen(center[0] + w // 2, center[1] + h // 2)
        STAGE += 1
    time.sleep(1.3)





