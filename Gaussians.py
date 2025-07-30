import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
 
img = cv.imread('lenna.bmp')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
assert img is not None, "file could not be read, check with os.path.exists()"

def dog(img):
    blur1 = cv.GaussianBlur(img,(5,5),0)
    blur2 = cv.GaussianBlur(img,(15,15),0)
    diff = cv.subtract(blur2, blur1)
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            if np.any(diff[i, j] > 8):
                diff[i, j] = [255, 255, 255]
    return diff

def edgeSmoothing(img):
    #smooth the edges of the image after applying dog
    kernel = np.ones((5, 5), np.float32) / 25
    smoothed = cv.filter2D(img, -1, kernel)
    return smoothed


diff = dog(img)
diff = edgeSmoothing(diff)


plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(diff),plt.title('Difference')
plt.xticks([]), plt.yticks([])
plt.show()