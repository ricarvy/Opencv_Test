# -*- coding: utf-8 -*-
# @Time    : 2018/1/12 17:03
# @Author  : Li Jiawei
# @FileName: haar_cascades.py
# @Software: PyCharm

### link: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html#face-detection
import numpy as np
import cv2

def rgb_bgr(img):
    (r,g,b)=cv2.split(img)
    return cv2.merge((b,g,r))

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')

img=cv2.imread('images/img5.jpg')
### img=rgb_bgr(img)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray,1.3,5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h,x:x+w]
    roi_color = img[y:y+h,x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


cv2.imshow('img',img)

cv2.waitKey()
cv2.destroyAllWindows()
