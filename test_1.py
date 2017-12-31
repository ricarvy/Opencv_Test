import cv2
print(cv2.__version__)
img=cv2.imread(filename='data/opencv_test_img.jpg',flags=cv2.IMREAD_GRAYSCALE)

### Resize the window
cv2.namedWindow('img',flags=cv2.WINDOW_NORMAL)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()