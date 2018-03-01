# -*- coding: utf-8 -*-
# @Time    : 2018/2/28 10:47
# @Author  : Li Jiawei
# @FileName: SIFT.py
# @Software: PyCharm

import cv2
import numpy as np

### default width of desscriptor histogram array
SIFT_DESCR_WIDTH = 4

### default number of bins per histogram in descriptor array
SIFT_DESCR_HIST_BINS = 8

### assumed Gaussian blur for input image
SIFT_INIT_SIGMA = 0.5

### maximum steps of keypoint interpolation(添写) before failure
SIFT_MAX_INTERP_STEPS=5

### width of border in which to ignore keypoints
SIFT_IMG_BORDER=5

### default number of bins in histogram for orientation assignment
SIFT_ORI_HIST_BINS = 36

### determines gaussians sigma for orientation assignment
SIFT_ORI_SIG_FCTR = 1.5

### determins the radius of region used in orientation assignment
SIFT_ORI_RADIUS = 3 * SIFT_ORI_SIG_FCTR

### orientation magnitude relative to max that results in new feature
SIFT_ORI_PEAK_RATIO = 0.8

### determines the size of a single descriptor orientation histogram
SIFT_DESCR_SCL_FCTR = 3.0

### threshold on magnitude of elements of descriptor vector
SIFT_DESCR_MAG_THR = 0.2

### factor used to convert floating-point descriptor to unsigned char
SIFT_INT_DESCT_FCTR = 512.0

### intermediate type used for DoG pyramids
SIFT_FIXPT_SCALE = 1

### test file name definition
SIFT_TEST_FILE='ali.jpg'

### 计算距离
def calculateDistance(row,col,x,y):
    row,col = np.floor(row/2), np.floor(col/2)
    distance=np.sqrt((row-x)**2 + (col-y)**2)
    return distance

### 计算高斯核元素
def calculateGaussianElement(distance,sigma):
    coefficient=1/(2 * np.pi * sigma**2)
    element=np.exp(-(distance**2/2*(sigma**2)))
    return coefficient * element

### 生成高斯核
def GaussianCoreGenerator(sigma):
    row=np.ceil(6*sigma+1).astype(np.int)
    col=np.ceil(6*sigma+1).astype(np.int)
    GaussianCore=np.zeros(shape=(row,col))
    for x in range(row):
        for y in range(col):
            distance=calculateDistance(row,col,x,y)
            element=calculateGaussianElement(distance,sigma)
            GaussianCore[x,y] = element
    return GaussianCore

### 使用高斯核进行高斯模糊
def GaussianBlurring(img,gaussianCore):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ### 矩阵拓展
    img_broad=np.zeros(shape=(img.shape[0]+np.floor(gaussianCore.shape[0]/2).astype(np.int)*2,img.shape[1]+np.floor(gaussianCore.shape[0]/2).astype(np.int)*2))

    print(img_broad.shape)

    # cv2.imshow('img',img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()



img=cv2.imread(filename=SIFT_TEST_FILE)
GaussianBlurring(img,GaussianCoreGenerator(0.6))