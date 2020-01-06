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
import ThrustRTC as trtc
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

print("==============LOADING IMAGE===============")

original = imread('/WD/PycharmProjects/BlobDetector/blobDetectionSequential/images/1.png')
grayscaleImage = rgb2gray(original)
# grayscaleImage = cv2.imread('/WD/PycharmProjects/BlobDetector/blobDetection/images/1.png', cv2.IMREAD_GRAYSCALE)
image = img_as_float(grayscaleImage)
# maxSigma = 100
# minSigma = 0.1
# sigmaRatio = 1.4
maxSigma = 100
minSigma = 0.1
sigmaRatio = 1.4
k = int(log(float(maxSigma) / minSigma, sigmaRatio)) + 1
sigmaList = np.array([minSigma + (sigmaRatio ** i) for i in range(k + 1)])
print("===============IMAGE LOADED===============")
octave = []
print("=========GENERATING BLURRED IMAGES========")
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
print("==========LOOKING FOR CANDIDATES==========")

mod = SourceModule("""
#include <stdio.h>
__inline__ __device__ float warpReduceMax(float val){   
    for (int offset = warpSize/2; offset > 0; offset/=2){
        float tmpVal = __shfl_down(val, offset );
        if (tmpVal > val){
            val = tmpVal;
        }
    }
    return val;
}

__global__ void findMax(float *xOut, float *yOut, float *zOut, float *in, int imageWidth, int imageHeight, int zIndex) { 
    float maxVal = 0;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;    
    int pixelIndex =  tz*imageWidth * imageHeight + imageWidth * (blockIdx.y + ty) + blockIdx.x + tx ;
    int pixelAmount = 3 * imageWidth * imageHeight;
    if(pixelIndex < pixelAmount){
        maxVal = max(maxVal, in[pixelIndex]);
        __syncthreads();
        maxVal = warpReduceMax(maxVal);
        __syncthreads();
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
            float expectedMax = in[pixelIndex + imageWidth*imageHeight + imageWidth + 1];
            float overflow1 = in[pixelIndex + imageWidth*imageHeight + imageWidth];
            float overflow2 = in[pixelIndex + imageWidth*imageHeight + imageWidth + 2];
            float overflow3 = in[pixelIndex + imageWidth*imageHeight - imageWidth + 1];
            float overflow4 = in[pixelIndex + imageWidth*imageHeight + 2*imageWidth + 1];
            if(expectedMax == maxVal && overflow1 != maxVal && overflow2 != maxVal && overflow3 != maxVal && overflow4 != maxVal){
                xOut[pixelIndex + imageWidth + 1] = blockIdx.x + 1  ;
                yOut[pixelIndex + imageWidth + 1] = blockIdx.y + 1 ;
                zOut[pixelIndex + imageWidth + 1] = zIndex;
            }
        }        
    }
}

__inline__ __device__ float warpReduceMin(float val){
    /*float tmpVal = __shfl_down(val, 3, 27);
    if (tmpVal < val){
        val = tmpVal;
    }*/
    for (int offset = warpSize; offset > 0; offset/=2){
        float tmpVal = __shfl_down(val, offset);
        if (tmpVal < val && tmpVal != 0){
            val = tmpVal;
        }
    }
    return val;
}

__global__ void findMin(float* xOut, float* yOut, float* zOut, float *in, int imageWidth, int imageHeight, int zIndex) {
    float minVal = 999;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int pixelIndex =  tz*imageWidth * imageHeight + imageWidth * (blockIdx.y + ty) + blockIdx.x + tx ;

    int pixelAmount = 3 * imageWidth * imageHeight;
    if(pixelIndex < pixelAmount){
        minVal = min(minVal, in[pixelIndex]);
        __syncthreads();
        minVal = warpReduceMin(minVal);
        __syncthreads();
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
            float expectedMin = in[pixelIndex + imageWidth*imageHeight + imageWidth + 1];
            float overflow1 = in[pixelIndex + imageWidth*imageHeight + imageWidth];
            float overflow2 = in[pixelIndex + imageWidth*imageHeight + imageWidth + 2];
            float overflow3 = in[pixelIndex + imageWidth*imageHeight - imageWidth + 1];
            float overflow4 = in[pixelIndex + imageWidth*imageHeight + 2*imageWidth + 1];
            if(expectedMin == minVal ){
                xOut[pixelIndex + imageWidth + 1] = blockIdx.x + 1  ;
                yOut[pixelIndex + imageWidth + 1] = blockIdx.y + 1 ;
                zOut[pixelIndex + imageWidth + 1] = zIndex;
            }
        }
    }
}
""")
newDogOctave = dogOctave.transpose(2, 0, 1)
gridSize = (newDogOctave.shape[2] - 2, newDogOctave.shape[1] - 2, 1)
tileWidth = 3
blockSize = (tileWidth, tileWidth, tileWidth)
xOutMax = np.zeros((21, 264306)).astype(np.float32)
yOutMax = np.zeros((21, 264306)).astype(np.float32)
zOutMax = np.zeros((21, 264306)).astype(np.float32)
findMax = mod.get_function("findMax")
for zIndex in range(1, newDogOctave.shape[0]-1):
    inputArray = newDogOctave[zIndex-1:zIndex+2, :, :].flatten().astype(np.float32)
    xOutTmp = np.zeros((264306,)).astype(np.float32)
    yOutTmp = np.zeros((264306,)).astype(np.float32)
    zOutTmp = np.zeros((264306,)).astype(np.float32)
    findMax(cuda.Out(xOutTmp), cuda.Out(yOutTmp), cuda.Out(zOutTmp), cuda.In(inputArray), np.int32(newDogOctave.shape[2]), np.int32(newDogOctave.shape[1]), np.int32(zIndex), block=blockSize, grid=gridSize)
    xOutMax[zIndex, :] = xOutTmp
    yOutMax[zIndex, :] = yOutTmp
    zOutMax[zIndex, :] = zOutTmp
xOutMax = xOutMax.astype(np.int32)
yOutMax = yOutMax.astype(np.int32)
zOutMax = zOutMax.astype(np.int32)

xOutMin = np.zeros((21, 264306)).astype(np.float32)
yOutMin = np.zeros((21, 264306)).astype(np.float32)
zOutMin = np.zeros((21, 264306)).astype(np.float32)
findMin = mod.get_function("findMin")
for zIndex in range(1, newDogOctave.shape[0]-1):
    inputArray = newDogOctave[zIndex-1:zIndex+2, :, :].flatten().astype(np.float32)
    xOutTmp = np.zeros((264306,)).astype(np.float32)
    yOutTmp = np.zeros((264306,)).astype(np.float32)
    zOutTmp = np.zeros((264306,)).astype(np.float32)
    findMin(cuda.Out(xOutTmp), cuda.Out(yOutTmp), cuda.Out(zOutTmp), cuda.In(inputArray), np.int32(newDogOctave.shape[2]), np.int32(newDogOctave.shape[1]), np.int32(zIndex), block=blockSize, grid=gridSize)
    xOutMin[zIndex, :] = xOutTmp
    yOutMin[zIndex, :] = yOutTmp
    zOutMin[zIndex, :] = zOutTmp
xOutMin = xOutMin.astype(np.int32)
yOutMin = yOutMin.astype(np.int32)
zOutMin = zOutMin.astype(np.int32)


# for i in range(w//2+1, dogOctave.shape[0]-w//2-1):
#     for j in range(w//2+1, dogOctave.shape[1]-w//2-1):
xOutMax = np.ma.masked_equal(xOutMax, 0).compressed()
yOutMax = np.ma.masked_equal(yOutMax, 0).compressed()
zOutMax = np.ma.masked_equal(zOutMax, 0).compressed()
for i in range(0, xOutMax.shape[0]):
    if xOutMax[i] != 0:
        valX = xOutMax[i]
        valY = yOutMax[i]
        valZ = zOutMax[i]
        candidates.append([valX, valY, valZ])
xOutMin = np.ma.masked_equal(xOutMin, 0).compressed()
yOutMin = np.ma.masked_equal(yOutMin, 0).compressed()
zOutMin = np.ma.masked_equal(zOutMin, 0).compressed()
for i in range(0, xOutMin.shape[0]):
    if xOutMin[i] != 0:
        valX = xOutMin[i]
        valY = yOutMin[i]
        valZ = zOutMin[i]
        candidates.append([valX, valY, valZ])
print("=====FOUND " + str(len(candidates)) + " CANDIDATES=====")
keypoints = []
r_th=10
t_c=0.03
R_th = (r_th+1)**2 / r_th

print("=====FILTERING CANDIDATES=====")
for candidate in candidates:
    y, x, s = candidate[1], candidate[0], candidate[2]
    dx = (dogOctave[y, x+1, s]-dogOctave[y, x-1, s])/2.
    dy = (dogOctave[y+1, x, s]-dogOctave[y-1, x, s])/2.
    ds = (dogOctave[y, x, s+1]-dogOctave[y, x, s-1])/2.

    dxx = dogOctave[y, x+1, s]-2*dogOctave[y, x, s]+dogOctave[y, x-1, s]
    dxy = ((dogOctave[y+1, x+1, s]-dogOctave[y+1, x-1, s]) - (dogOctave[y-1, x+1, s]-dogOctave[y-1, x-1, s]))/4.
    dxs = ((dogOctave[y, x+1, s+1]-dogOctave[y, x-1, s+1]) - (dogOctave[y, x+1, s-1]-dogOctave[y, x-1, s-1]))/4.
    dyy = dogOctave[y+1, x, s]-2*dogOctave[y, x, s]+dogOctave[y-1, x, s]
    dys = ((dogOctave[y+1, x, s+1]-dogOctave[y-1, x, s+1]) - (dogOctave[y+1, x, s-1]-dogOctave[y-1, x, s-1]))/4.
    dss = dogOctave[y, x, s+1]-2*dogOctave[y, x, s]+dogOctave[y, x, s-1]

    J = np.array([dx, dy, ds])
    HD = np.array([
        [dxx, dxy, dxs],
        [dxy, dyy, dys],
        [dxs, dys, dss]])

    offset = -LA.inv(HD).dot(J)
    contrast = dogOctave[y, x, s] + .5*J.dot(offset)
    if abs(contrast) < t_c:
        continue

    w, v = LA.eig(HD)
    r = w[1]/w[0]
    R = (r+1)**2 / r
    if R > R_th:
        continue

    kp = np.array([x, y, s]) + offset
    if kp[1] >= dogOctave.shape[0] or kp[0] >= dogOctave.shape[1]:
        continue

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

