import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math
 
img = cv.imread('oil_rig.JPG')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = cv.resize(img, (800, 600))  # Resize for better visibility
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
    kernel1, kernel2 = generateGaussianKernel(sig, l), generateGaussianKernel(sig, l)
    #apply kernel through convolution
    blur1 = cv.filter2D(img, -1, kernel1)
    blur2 = cv.filter2D(img, -1, kernel2)
    return blur1, blur2

def blurTaoist(img, sig = 1., l = 5, tao = 2.):
    kernel1, kernel2 = generateGaussianKernel((1 + tao) * sig, l), generateGaussianKernel(sig * tao, l)
    blur1 = cv.filter2D(img, -1, kernel1)
    blur2 = cv.filter2D(img, -1, kernel2)
    return blur1, blur2

def structureTensor(img, sigma=1.0):
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # Compute image gradients
    Ix = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    Iy = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

    Ixx = cv.GaussianBlur(Ix * Ix, (0, 0), sigma)
    Ixy = cv.GaussianBlur(Ix * Iy, (0, 0), sigma)
    Iyy = cv.GaussianBlur(Iy * Iy, (0, 0), sigma)

    h, w = gray.shape
    eigvecs = np.zeros((h, w, 2), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            S = np.array([[Ixx[y, x], Ixy[y, x]],
                          [Ixy[y, x], Iyy[y, x]]])
            eigvals, eigvec = np.linalg.eigh(S)
            dominant = eigvec[:, np.argmax(eigvals)]
            eigvecs[y, x] = dominant

    return eigvecs 
def ETFBlur(img, etf, radius=5):
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    h, w = gray.shape
    blurred = np.zeros_like(gray, dtype=np.float32)

    for y in range(h):
        for x in range(w):
            acc = 0.0
            weight_sum = 0.0

            for s in range(-radius, radius + 1):
                dx, dy = etf[y, x]
                norm = np.sqrt(dx*dx + dy*dy) + 1e-6
                dir_x = dx / norm
                dir_y = dy / norm

                sample_x = int(round(x + s * dir_x))
                sample_y = int(round(y + s * dir_y))

                if 0 <= sample_x < w and 0 <= sample_y < h:
                    weight = np.exp(-s*s / (2 * (radius/2)**2)) 
                    acc += gray[sample_y, sample_x] * weight
                    weight_sum += weight

            blurred[y, x] = acc / weight_sum if weight_sum > 0 else gray[y, x]

    return np.uint8(np.clip(blurred, 0, 255))


def dogManual(img, sig = 1., l = 5, threshold = 8, phi = 1.):
    blur1, blur2 = blurManual(img, sig, l)
    diff = cv.absdiff(blur2, blur1)
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            if np.any(diff[i, j] > threshold):
                diff[i, j] = 1
            else:
                diff[i,j] = 1 + np.tanh(phi * (diff[i, j] - threshold))
    return diff

def germanDog(img, sig = 1., l = 5, threshold = 8, tao = 2., phi = 1.):
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    blur1, blur2 = blurTaoist(gray, sig, l, tao)
    diff = cv.absdiff(blur1, blur2).astype(np.float32)  # promote to float for tanh
    result = np.zeros_like(diff, dtype=np.uint8)

    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            if np.any(diff[i, j] > threshold):
                result[i, j] = 255
            else:
                val = 1 + np.tanh(phi * (diff[i, j] - tao))
                val = np.clip(val * 127.5, 0, 255)
                result[i, j] = val.astype(np.uint8)
    return result


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
    taoRaw = cv.getTrackbarPos('Tao x10', window)
    phiRaw = cv.getTrackbarPos('Phi x10', window)

    # Ensure odd kernel size >= 3
    if ksize % 2 == 0:
        ksize += 1
    if ksize < 3:
        ksize = 3

    sigma = sigma_raw / 10.0
    tao = max(0.1, taoRaw / 10.0)
    phi = max(0.1, phiRaw / 10.0)
    print(phi)
    #result = germanDog(img, sig=sigma, l=ksize, threshold=threshold, tao = tao, phi = phi)
    etf = structureTensor(img)  
    result = ETFBlur(img, etf, radius=5)

    
    if smooth_toggle:
        result = edgeSmoothing(result)

    # Convert RGB back to BGR for OpenCV display
    cv.imshow(window, cv.cvtColor(result, cv.COLOR_RGB2BGR))

# Setup window and trackbars
window = 'Dog'
cv.namedWindow(window)

cv.createTrackbar('Threshold', window, 8, 255, onChange)
cv.createTrackbar('Sigma x10', window, 10, 50, onChange)     # Sigma from 0.1 to 5.0
cv.createTrackbar('Kernel Size', window, 5, 31, onChange)    # Kernel size from 3 to 31
cv.createTrackbar('Edge Smoothing', window, 0, 1, onChange)  # 0 = off, 1 = on
cv.createTrackbar('Tao x10', window, 10, 50, onChange) 
cv.createTrackbar('Phi x10', window, 10, 100, onChange) 

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
