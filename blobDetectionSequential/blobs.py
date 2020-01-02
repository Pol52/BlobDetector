from skimage.color import rgb2gray
import numpy as np
from math import log
from skimage import io
from scipy.ndimage import gaussian_filter
from skimage.io import imread
from skimage import img_as_float
import numpy.linalg as LA
import cv2
import math

print("======LOADING IMAGE======")

original = imread('/WD/PycharmProjects/BlobDetector/blobDetectionSequential/images/1.png')
grayscaleImage = rgb2gray(original)
# grayscaleImage = cv2.imread('/WD/PycharmProjects/BlobDetector/blobDetection/images/1.png', cv2.IMREAD_GRAYSCALE)
image = img_as_float(grayscaleImage)
maxSigma = 100
minSigma = 0.1
sigmaRatio = 1.4
k = int(log(float(maxSigma) / minSigma, sigmaRatio)) + 1
sigmaList = np.array([minSigma + (sigmaRatio ** i) for i in range(k + 1)])
print("======IMAGE LOADED======")
octave = []
print("======GENERATING BLURRED IMAGES=====")
for sigma in sigmaList:
    blurredImage = gaussian_filter(image, sigma)
    # blurredImage = cv2.GaussianBlur(image, (33,33), sigmaX=sigma, borderType = cv2.BORDER_DEFAULT)
    octave.append(blurredImage)
tempDogOctave = []
print("=====BUILDING DIFFERENCE OF GAUSSIANS=====")
for index in range(1, len(octave)):
    dog = octave[index] - octave[index-1]
    # io.imsave(str(index) + ".jpg", dog)
    tempDogOctave.append(dog)
dogOctave = np.concatenate([o[:, :, np.newaxis] for o in tempDogOctave], axis=2)
candidates = []
print("=====LOOKING FOR CANDIDATES=====")
# for i in range(w//2+1, dogOctave.shape[0]-w//2-1):
#     for j in range(w//2+1, dogOctave.shape[1]-w//2-1):
for i in range(1, dogOctave.shape[0]-1):
    for j in range(1, dogOctave.shape[1]-1):
        for k in range(1, dogOctave.shape[2]-1):
            patch = dogOctave[i-1:i+2, j-1:j+2, k-1:k+2]
            if np.argmax(patch) == 13 or np.argmin(patch) == 13:
                candidates.append([i, j, k])
print("=====FOUND " + str(len(candidates)) + " CANDIDATES=====")
keypoints = []
r_th=10
t_c=0.03
R_th = (r_th+1)**2 / r_th
print("=====FILTERING CANDIDATES=====")
for candidate in candidates:
    y, x, s = candidate[0], candidate[1], candidate[2]
    dx = (dogOctave[y,x+1,s]-dogOctave[y,x-1,s])/2.
    dy = (dogOctave[y+1,x,s]-dogOctave[y-1,x,s])/2.
    ds = (dogOctave[y,x,s+1]-dogOctave[y,x,s-1])/2.

    dxx = dogOctave[y,x+1,s]-2*dogOctave[y,x,s]+dogOctave[y,x-1,s]
    dxy = ((dogOctave[y+1,x+1,s]-dogOctave[y+1,x-1,s]) - (dogOctave[y-1,x+1,s]-dogOctave[y-1,x-1,s]))/4.
    dxs = ((dogOctave[y,x+1,s+1]-dogOctave[y,x-1,s+1]) - (dogOctave[y,x+1,s-1]-dogOctave[y,x-1,s-1]))/4.
    dyy = dogOctave[y+1,x,s]-2*dogOctave[y,x,s]+dogOctave[y-1,x,s]
    dys = ((dogOctave[y+1,x,s+1]-dogOctave[y-1,x,s+1]) - (dogOctave[y+1,x,s-1]-dogOctave[y-1,x,s-1]))/4.
    dss = dogOctave[y,x,s+1]-2*dogOctave[y,x,s]+dogOctave[y,x,s-1]

    J = np.array([dx, dy, ds])
    HD = np.array([
        [dxx, dxy, dxs],
        [dxy, dyy, dys],
        [dxs, dys, dss]])

    offset = -LA.inv(HD).dot(J)
    contrast = dogOctave[y,x,s] + .5*J.dot(offset)
    if abs(contrast) < t_c: continue

    w, v = LA.eig(HD)
    r = w[1]/w[0]
    R = (r+1)**2 / r
    if R > R_th: continue

    kp = np.array([x, y, s]) + offset
    if kp[1] >= dogOctave.shape[0] or kp[0] >= dogOctave.shape[1]: continue

    keypoints.append(kp)
# io.imshow(dogOctave[1], cmap='gray')
print("=====REMAINING KEYPOINTS = " + str(len(keypoints)) + "=====")
resultImage = cv2.imread('/WD/PycharmProjects/BlobDetector/blobDetectionSequential/images/1.png', cv2.COLOR_RGB2BGR)
index = 1
for keypoint in keypoints:#keypoint sono x,y,candidates sono y,x
    scalespace = int(np.ceil(keypoint[2]))
    if scalespace >= 0:
        radius = math.sqrt(2)*scalespace*1.6
        resultImage = cv2.circle(resultImage, (int(keypoint[0]), int(keypoint[1])), int(radius),(0,255,0), 2)

cv2.imshow('Result', resultImage)
cv2.waitKey(0)
