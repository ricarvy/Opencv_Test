# -*- coding: utf-8 -*-
# @Time    : 2018/1/12 18:47
# @Author  : Li Jiawei
# @FileName: __init__.py.py
# @Software: PyCharm

### link: http://www.cnblogs.com/neo-T/p/6426029.html
import cv2
import sys
import PIL as Image

# def catchUsbVideo(window_name,camera_idx):
#     cv2.namedWindow(winname=window_name)
#     cap=cv2.VideoCapture(camera_idx)
#     classifier=cv2.CascadeClassifier('../Algorithm/Object_Detection/data/haarcascade_eye.xml')
#     color=(0,255,0)
#     while cap.isOpened():
#         ok,frame=cap.read()
#         if not ok:
#             break
#         gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#         faceRects=classifier.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=3,minSize=(32,32))
#         if len(faceRects) > 0:
#             for faceRect in faceRects:
#                 x,y,w,h = faceRect
#                 cv2.rectangle(frame,(x-10,y-10),(x+w+10,y+h+10),color,2)
#
#         cv2.imshow(window_name,frame)
#         c=cv2.waitKey(10)
#         if c == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
# catchUsbVideo('captureWindows',0)








def catchUsbVideo(winname,device_index):
    cv2.namedWindow(winname=winname)
    cap=cv2.VideoCapture(device_index)

    while cap.isOpened():
        _,frame=cap.read()
        clf=cv2.CascadeClassifier('../Algorithm/Object_Detection/data/haarcascade_mcs_nose.xml')
        faceRects=clf.detectMultiScale(image=frame,scaleFactor=1.2,minNeighbors=3,minSize=(32,32))
        for faceRect in faceRects:
            x,y,w,h=faceRect
            cv2.rectangle(frame,(x-10,y-10),(x+w+10,y+h+10),color=(0,255,0),thickness=2)
            cv2.putText(frame,'me',(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),4)


        cv2.imshow(winname,frame)

        k=cv2.waitKey(10)
        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


catchUsbVideo(winname='test',device_index=0)








