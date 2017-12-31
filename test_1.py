import cv2
import matplotlib.pyplot as plt
print(cv2.__version__)

### Read an image
img=cv2.imread(filename='data/ali.jpg')

### Resize the window
# cv2.namedWindow('img',flags=cv2.WINDOW_NORMAL)
cv2.imshow('img',img)

### show the image using plt
# plt.imshow(img)
# plt.xticks([]),plt.yticks([])
# plt.show()

### receive the keyboard event
k=cv2.waitKey(0)
print(k)
if k == ord('s'):
    ### Write an image
    ### save the handled image to working directory
    cv2.imwrite('img_gray.png', img=img)
    cv2.destroyAllWindows()
if k == 27:
    cv2.destroyAllWindows()


