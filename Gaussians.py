import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math
 
img = cv.imread('oil_rig.JPG')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = cv.resize(img, (800, 600))  # Resize for better visibility
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
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

def convertColor(img):
    if len(img.shape) == 3:
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        return gray
    return img

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


sigmaOld = 0
Ix = None
Ixy = None
def structureTensor(gray, sigma=1.0):
    global sigmaOld, Ix, Iy
    if(sigmaOld != sigma):
        sigmaOld = sigma
        Ix = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
        Iy = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
    if(Ix is None or Iy is None):
        Ix = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
        Iy = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
    ksize = max(3, int(2 * round(3 * sigma) + 1))  # must be odd

    Ixx = cv.GaussianBlur(Ix * Ix, (ksize, ksize), sigma)
    Ixy = cv.GaussianBlur(Ix * Iy, (ksize, ksize), sigma)
    Iyy = cv.GaussianBlur(Iy * Iy, (ksize, ksize), sigma)

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
etf = None
def blurTaoist(img, sig = 1., l = 5, tao = 2.):
    global etf, sigmaOld, gray
    if etf is None or sigmaOld != sig:
        sigmaOld = sig
        etf = structureTensor(gray, sigma=sig)
        print("recalculated ETF with sigma:", sig)
    print("blurred taoist")
  #  kernel1, kernel2 = generateGaussianKernel((1 + tao) * sig, l), generateGaussianKernel(sig * tao, l)
  #  blur1 = cv.filter2D(img, -1, kernel1)
    blur1 = ETFBlur(img, etf, radius=5)
   # blur2 = cv.filter2D(img, -1, kernel2)
    blur2 = ETFBlur(img, etf, radius=10)
    return blur1, blur2

def ETFBlur(img, etf, radius=5):
    h, w = gray.shape
    blurred = np.zeros_like(gray, dtype=np.float32)

    for y in range(h):
        for x in range(w):
            acc = 0.0
            weight_sum = 0.0

            dx, dy = etf[y, x]
            norm = np.sqrt(dx*dx + dy*dy) + 1e-6
            dir_x = dx / norm
            dir_y = dy / norm
            weights = np.exp(-np.square(np.arange(-radius, radius+1)) / (2 * (radius/2)**2))
            for s in range(-radius, radius + 1):

                sample_x = int(round(x + s * dir_x))
                sample_y = int(round(y + s * dir_y))

                if 0 <= sample_x < w and 0 <= sample_y < h:
                    weight = weights[s + radius]
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



def germanDog(img, etf, gray, sig = 1., l = 5, threshold = 8, tao = 2., phi = 1.):
    blur1, blur2 = blurTaoist(gray, sig, l, tao)
    diff = cv.absdiff(blur1, blur2).astype(np.float32)  # promote to float for tanh
    result = np.zeros_like(diff, dtype=np.uint8)
    
    '''
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            if np.any(diff[i, j] > threshold):
                result[i, j] = 255
            else:
                val = 1 + np.tanh(phi * (diff[i, j] - tao))
                val = np.clip(val * 127.5, 0, 255)
                result[i, j] = val.astype(np.uint8)
              
    # vectorized is just better
    mask = diff > threshold
    tanh_vals = np.tanh(phi * (diff - threshold)).astype(np.float32)
    print("Diff min/max:", diff.min(), diff.max())
    print(tanh_vals)
    window3 = "Tanh Values"
    cv.imshow(window3, cv.cvtColor(tanh_vals, cv.COLOR_GRAY2BGR))
    normalized = ((tanh_vals + 1) / 2) * 255 
    window2 = "Normalized Difference"
    cv.imshow(window2, cv.cvtColor(normalized, cv.COLOR_GRAY2BGR))
    window4 = "Mask"
    cv.imshow(window4, (mask.astype(np.uint8) * 255))
    print(normalized)
    result = np.where(mask, 255, normalized).astype(np.uint8)
    return result
    '''
    tanhInput = diff - threshold
    tanhVals = 1 + np.tanh(phi * tanhInput)
    tanhScaled = (tanhVals * 127.5).clip(0, 255)
    mask = diff >= threshold
    white = np.full_like(tanhScaled, 255, dtype=np.float32)
    result = np.where(mask, white, tanhScaled).astype(np.uint8)



    cv.imshow("Normalized Difference", (diff / diff.max() * 255).astype(np.uint8))
    cv.imshow("Tanh Values", tanhScaled.astype(np.uint8))
    cv.imshow("Mask", (mask * 255).astype(np.uint8))  # This should now be mostly white

    return result


def edgeSmoothing(img):
    #smooth the edges of the image after applying dog
    kernel = np.ones((5, 5), np.float32) / 25
    smoothed = cv.filter2D(img, -1, kernel)
    return smoothed
counter = 0;
def onChange(val):
    global etf, sigmaOld, Ix, Ixy, window, gray, counter
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

    sigma = max(0.1, sigma_raw / 10.0)
    tao = max(0.1, taoRaw / 10.0)
    phi = max(0.1, phiRaw / 10.0)
    print(phi)
    if(counter > 5):
        result = germanDog(img, etf, gray, sig=sigma, l=ksize, threshold=threshold, tao = tao, phi = phi)
        
        if smooth_toggle:
            result = edgeSmoothing(result)
        

        if result.ndim == 2:
            cv.imshow(window, result)
        else:
            cv.imshow(window, cv.cvtColor(result, cv.COLOR_RGB2BGR))

    counter += 1

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
