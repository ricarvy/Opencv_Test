# -*- coding: utf-8 -*-
# @Time    : 2018/1/2 13:03
# @Author  : Li Jiawei
# @FileName: SIFT.py
# @Software: PyCharm

import cv2

def imagePreprocessing(image_name):
    img=cv2.imread(filename=image_name,flags=cv2.COLOR_BGR2GRAY)
    return img

def convertImageWithGaussianBlur(img):
    ### define the kernel size
    kernel=(7,7)
    sigma_X=3
    img=cv2.GaussianBlur(src=img,ksize=kernel,sigmaX=sigma_X)
    return img

def wait():
    cv2.waitKey()

def finish():
    cv2.destroyAllWindows()

def SIFT():
    #sift=cv2.SIFT()
    pass


def main():
    image_name='ali.jpg'
    img=imagePreprocessing(image_name=image_name)
    img=convertImageWithGaussianBlur(img)

    SIFT()
    wait()
    finish()

main()