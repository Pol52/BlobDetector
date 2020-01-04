# import pycuda.autoinit
# import pycuda.driver as drv
# import numpy
#
# from pycuda.compiler import SourceModule
# mod = SourceModule("""
# __global__ void multiply_them(float *dest, float *a, float *b)
# {
#   const int i = threadIdx.x;
#   dest[i] = a[i] * b[i];
# }
# """)
#
# multiply_them = mod.get_function("multiply_them")
#
# a = numpy.random.randn(400).astype(numpy.float32)
# b = numpy.random.randn(400).astype(numpy.float32)
#
# dest = numpy.zeros_like(a)
# multiply_them(
#     drv.Out(dest), drv.In(a), drv.In(b),
#     block=(400,1,1), grid=(1,1))
#
# print(dest)

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
import sys

# numpy.set_printoptions(threshold=sys.siz)

print("==============LOADING IMAGE===============")

original = imread('/WD/PycharmProjects/BlobDetector/blobDetectionSequential/images/1.png')
grayscaleImage = rgb2gray(original)
# grayscaleImage = cv2.imread('/WD/PycharmProjects/BlobDetector/blobDetection/images/1.png', cv2.IMREAD_GRAYSCALE)
image = img_as_float(grayscaleImage)
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

__global__ void findMax(float *out, float *in, int imageWidth, int imageHeight) { 
    float minVal = 0;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;    
    int pixelIndex =  tz*imageWidth * imageHeight + imageWidth * (blockIdx.y + ty) + blockIdx.x + tx ;
    int pixelAmount = 3 * imageWidth * imageHeight;
    if(pixelIndex < pixelAmount){
        minVal = max(minVal, in[pixelIndex]);
        __syncthreads();
        minVal = warpReduceMax(minVal);
        __syncthreads();
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
            printf("Block %d.%d value: %e\\n", blockIdx.x, blockIdx.y, minVal);
        }
        out[pixelIndex] = minVal;
    }
}

__inline__ __device__ float warpReduceMin(float val){   
    float tmpVal = __shfl_down(val, 3, 27);
    if (tmpVal < val){
        val = tmpVal;
    }
    for (int offset = 12; offset > 0; offset/=2){
        float tmpVal = __shfl_down(val, offset, offset*2 );
        if (tmpVal < val){
            val = tmpVal;
        }
    }
    return val;
}

__global__ void findMin(float *out, float *in) { 
    float minVal = 999;
    int imageWidth = 4;
    int imageHeight = 4;
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
            printf("Block %d.%d value: %e\\n", blockIdx.x, blockIdx.y, minVal);
        }
        out[pixelIndex] = minVal;
    }
}
""")

input = dogOctave.transpose(2, 0, 1)[0, :, :].flatten().astype(np.float32)
dest = np.zeros_like(input).astype(np.float32)
tileWidth = 3
blockSize = (tileWidth, tileWidth, tileWidth)
dogOctave = np.random.rand(3,4,4).astype(np.float32)
print(dogOctave)
input = dogOctave.flatten().astype(np.float32)
dest = np.zeros_like(input).astype(np.float32)
gridSize = (dogOctave.shape[2] - 2, dogOctave.shape[1] - 2, 1) #PER DOG OCTAVE ORIGINALE SONO shape[1] e shape[0]
findMin = mod.get_function("findMin")
findMin(cuda.Out(dest), cuda.In(input), block=blockSize, grid=gridSize)
# gridSize = (int(np.ceil(dogOctave.shape[1]/tileWidth)), int(np.ceil(dogOctave.shape[0]/tileWidth)), 1)
#findMax = mod.get_function("findMax")
#findMax(cuda.Out(dest), cuda.In(input), np.int32(dogOctave.shape[1]), np.int32(dogOctave.shape[1]), block=blockSize, grid=gridSize)

cuda.Context.synchronize()
print(dest[0])
print(dest[1])
print(dest[2])
print(dest[3])
print(dest[4])
print(dest[5])
# # for i in range(w//2+1, dogOctave.shape[0]-w//2-1):
# #     for j in range(w//2+1, dogOctave.shape[1]-w//2-1):
# # for i in range(1, dogOctave.shape[0]-1):
# #     for j in range(1, dogOctave.shape[1]-1):
# #         for k in range(1, dogOctave.shape[2]-1):
# #             patch = dogOctave[i-1:i+2, j-1:j+2, k-1:k+2]
# #             if np.argmax(patch) == 13 or np.argmin(patch) == 13:
# #                 candidates.append([i, j, k])
# # devOctave = trtc.device_vector_from_numpy(dogOctave)
# # print(devOctave.to_host()[0])
# # print("=====FOUND " + str(len(candidates)) + " CANDIDATES=====")
# # keypoints = []
# # r_th=10
# # t_c=0.03
# # R_th = (r_th+1)**2 / r_th
# # print("=====FILTERING CANDIDATES=====")
# # for candidate in candidates:
# #     y, x, s = candidate[0], candidate[1], candidate[2]
# #     dx = (dogOctave[y,x+1,s]-dogOctave[y,x-1,s])/2.
# #     dy = (dogOctave[y+1,x,s]-dogOctave[y-1,x,s])/2.
# #     ds = (dogOctave[y,x,s+1]-dogOctave[y,x,s-1])/2.
# #
# #     dxx = dogOctave[y,x+1,s]-2*dogOctave[y,x,s]+dogOctave[y,x-1,s]
# #     dxy = ((dogOctave[y+1,x+1,s]-dogOctave[y+1,x-1,s]) - (dogOctave[y-1,x+1,s]-dogOctave[y-1,x-1,s]))/4.
# #     dxs = ((dogOctave[y,x+1,s+1]-dogOctave[y,x-1,s+1]) - (dogOctave[y,x+1,s-1]-dogOctave[y,x-1,s-1]))/4.
# #     dyy = dogOctave[y+1,x,s]-2*dogOctave[y,x,s]+dogOctave[y-1,x,s]
# #     dys = ((dogOctave[y+1,x,s+1]-dogOctave[y-1,x,s+1]) - (dogOctave[y+1,x,s-1]-dogOctave[y-1,x,s-1]))/4.
# #     dss = dogOctave[y,x,s+1]-2*dogOctave[y,x,s]+dogOctave[y,x,s-1]
# #
# #     J = np.array([dx, dy, ds])
# #     HD = np.array([
# #         [dxx, dxy, dxs],
# #         [dxy, dyy, dys],
# #         [dxs, dys, dss]])
# #
# #     offset = -LA.inv(HD).dot(J)
# #     contrast = dogOctave[y,x,s] + .5*J.dot(offset)
# #     if abs(contrast) < t_c: continue
# #
# #     w, v = LA.eig(HD)
# #     r = w[1]/w[0]
# #     R = (r+1)**2 / r
# #     if R > R_th: continue
# #
# #     kp = np.array([x, y, s]) + offset
# #     if kp[1] >= dogOctave.shape[0] or kp[0] >= dogOctave.shape[1]: continue
# #
# #     keypoints.append(kp)
# # # io.imshow(dogOctave[1], cmap='gray')
# # print("=====REMAINING KEYPOINTS = " + str(len(keypoints)) + "=====")
# # resultImage = cv2.imread('/WD/PycharmProjects/BlobDetector/blobDetectionSequential/images/1.png', cv2.COLOR_RGB2BGR)
# # index = 1
# # for keypoint in keypoints:#keypoint sono x,y,candidates sono y,x
# #     scalespace = int(np.ceil(keypoint[2]))
# #     if scalespace >= 0:
# #         radius = math.sqrt(2)*scalespace*1.6
# #         resultImage = cv2.circle(resultImage, (int(keypoint[0]), int(keypoint[1])), int(radius),(0,255,0), 2)
# #
# # cv2.imshow('Result', resultImage)
# # cv2.waitKey(0)

# /*FUNZIONANTE 2D
# __inline__ __device__ float warpReduceMin(float val)
# {
#     float tmpVal = __shfl_down(val, 1, 9);
# if (tmpVal < val)
# {
#     val = tmpVal;
# }
# for (int offset = 4; offset > 0; offset/=2)
# {
# //float tmpVal = __shfl_down(val, offset);
# float tmpVal = __shfl_down(val, offset);
# if (tmpVal < val)
# {
# val = tmpVal;
# }
# }
# return val;
# }
#
# __global__ void findMin(float *out, float *in) {
# float minVal = 0;
# int imageWidth = 3;
# int tx = threadIdx.x;
# int ty = threadIdx.y;
# int pixelIndex = imageWidth * (blockIdx.y + ty) + blockIdx.x + tx ;
# if(pixelIndex <= 25){
# minVal = max(minVal, in[pixelIndex]);
# __syncthreads();
# minVal = warpReduceMin(minVal);
# if(threadIdx.x == 0 && threadIdx.y == 0){
# printf("Block %d.%d value: %e\\n", blockIdx.x, blockIdx.y, minVal);
# }
# __syncthreads();
#
# out[pixelIndex] = minVal;
# }
# }*/