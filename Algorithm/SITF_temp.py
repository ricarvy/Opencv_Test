# -*- coding: utf-8 -*-
# @Time    : 2018/1/2 17:00
# @Author  : Li Jiawei
# @FileName: SITF_temp.py
# @Software: PyCharm

### link: http://blog.csdn.net/Eddy_zheng/article/details/78916009

from lake.decorator import time_cost
import cv2

def bgr_rgb(img):
    (r,g,b)=cv2.split(img)
    return cv2.merge([b,g,r])

def orb_detect(image_a,image_b):
    orb=cv2.ORB_create()
    kp1,des1=orb.detectAndCompute(image_a,None)
    kp2,des2=orb.detectAndCompute(image_b,None)

    bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

    matches=bf.match(des1,des2)

    matches=sorted(matches,key=lambda x:x.distance)

    img3=cv2.drawMatches(image_a,kp1,image_b,kp2,matches[:10],None,flags=2)

    return bgr_rgb(img3)

def sift_detect(img1,img2,detector='surf'):
    if detector.startswith('si'):
        print('sift detector')
        sift=cv2.haarcascades
    else:
        print('suft detector')
        surf=cv2.SURF_create()


if __name__ == "__main__":
    image_a=cv2.imread('../data/opencv_test_img.jpg',0)
    image_b = cv2.imread('../data/opencv_test_img_50.jpg', 0)

    img=orb_detect(image_a,image_b)
    cv2.namedWindow('img',cv2.WINDOW_NORMAL)
    cv2.imshow('img',img)

    cv2.waitKey()
    cv2.destroyAllWindows()

