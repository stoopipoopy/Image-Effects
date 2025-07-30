import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
 
img = cv.imread('lenna.bmp')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
assert img is not None, "file could not be read, check with os.path.exists()"


# Difference of Gaussians using openCV
def dog(img):
    blur1 = cv.GaussianBlur(img,(5,5),0)
    blur2 = cv.GaussianBlur(img,(15,15),0)
    diff = cv.subtract(blur2, blur1)
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            if np.any(diff[i, j] > 8):
                diff[i, j] = [255, 255, 255]
    return diff

def generateGaussianKernel(sig, l):
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    kernel = kernel / np.sum(kernel)
    return kernel

def blurManual(img, sig = 1., l = 5):
    kernel1, kernel2 = generateGaussianKernel(sig, l), generateGaussianKernel(sig * 2, l)
    #apply kernel through convolution
    blur1 = cv.filter2D(img, -1, kernel1)
    blur2 = cv.filter2D(img, -1, kernel2)
    return blur1, blur2

def dogManual(img, sig = 1., l = 5, threshold = 8):
    blur1, blur2 = blurManual(img, sig, l)
    diff = cv.absdiff(blur2, blur1)
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            if np.any(diff[i, j] > threshold):
                diff[i, j] = [255, 255, 255]
    return diff

def edgeSmoothing(img):
    #smooth the edges of the image after applying dog
    kernel = np.ones((5, 5), np.float32) / 25
    smoothed = cv.filter2D(img, -1, kernel)
    return smoothed

def onChange(val):
    threshold = cv.getTrackbarPos('Threshold', window)
    sigma_raw = cv.getTrackbarPos('Sigma x10', window)
    ksize = cv.getTrackbarPos('Kernel Size', window)
    smooth_toggle = cv.getTrackbarPos('Edge Smoothing', window)

    # Ensure odd kernel size >= 3
    if ksize % 2 == 0:
        ksize += 1
    if ksize < 3:
        ksize = 3

    sigma = sigma_raw / 10.0
    result = dogManual(img, sig=sigma, l=ksize, threshold=threshold)

    if smooth_toggle:
        result = edgeSmoothing(result)

    # Convert RGB back to BGR for OpenCV display
    cv.imshow(window, cv.cvtColor(result, cv.COLOR_RGB2BGR))

# Setup window and trackbars
window = 'DoG Edge Detector'
cv.namedWindow(window)

cv.createTrackbar('Threshold', window, 8, 255, onChange)
cv.createTrackbar('Sigma x10', window, 10, 50, onChange)     # Sigma from 0.1 to 5.0
cv.createTrackbar('Kernel Size', window, 5, 31, onChange)    # Kernel size from 3 to 31
cv.createTrackbar('Edge Smoothing', window, 0, 1, onChange)  # 0 = off, 1 = on

onChange(8)
cv.waitKey(0)
cv.destroyAllWindows()


'''
diff = dogManual(img)
blur1, blur2 = blurManual(img)
#diff = edgeSmoothing(diff)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(diff),plt.title('Difference')
plt.xticks([]), plt.yticks([])
plt.show()
'''
