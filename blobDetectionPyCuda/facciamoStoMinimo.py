from math import log

import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
from scipy.ndimage import gaussian_filter
from skimage import img_as_float
from skimage.color import rgb2gray
from skimage.io import imread

print(pycuda.autoinit.context.set_limit(cuda.limit.PRINTF_FIFO_SIZE, 20000000))
print(pycuda.autoinit.context.get_limit(cuda.limit.PRINTF_FIFO_SIZE))
mod = SourceModule("""
#include <stdio.h>

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
                printf("kp");
                xOut[pixelIndex + imageWidth + 1] = blockIdx.x + 1  ;
                yOut[pixelIndex + imageWidth + 1] = blockIdx.y + 1 ;
                zOut[pixelIndex + imageWidth + 1] = zIndex;
            }
        }
    }
}
""")
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
newDogOctave = dogOctave.transpose(2, 0, 1)
gridSize = (newDogOctave.shape[2] - 2, newDogOctave.shape[1] - 2, 1)
tileWidth = 3
blockSize = (tileWidth, tileWidth, tileWidth)
findMin = mod.get_function("findMin")
k = 19
inputArray = newDogOctave[k-1:k+2, :, :].flatten().astype(np.float32)
xOutMin = np.zeros((264306,)).astype(np.float32)
yOutMin = np.zeros((264306,)).astype(np.float32)
zOutMin = np.zeros((264306,)).astype(np.float32)
findMin(cuda.Out(xOutMin), cuda.Out(yOutMin), cuda.Out(zOutMin),
        cuda.In(inputArray),
        np.int32(newDogOctave.shape[2]),
        np.int32(newDogOctave.shape[1]),
        np.int32(k),
        block=blockSize, grid=gridSize)
cuda.Context.synchronize()
xOutMin = xOutMin.astype(np.int32)
yOutMin = yOutMin.astype(np.int32)
zOutMin = zOutMin.astype(np.int32)
xOutMin = np.ma.masked_equal(xOutMin, 0).compressed()
yOutMin = np.ma.masked_equal(yOutMin, 0).compressed()
zOutMin = np.ma.masked_equal(zOutMin, 0).compressed()
total = 0

for i in range(1, newDogOctave.shape[1]-1):
    for j in range(1, newDogOctave.shape[2]-1):
        patch = newDogOctave[k-1:k+2, i-1:i+2, j-1:j+2]
        if np.argmin(patch) == 13:
            total += 1
            candidates.append([j,i,k])

print("totale")
print(total)

for candidateIndex in range(len(candidates)):
    if xOutMin[candidateIndex] != candidates[candidateIndex][0] or yOutMin[candidateIndex] != candidates[candidateIndex][1] or zOutMin[candidateIndex] != candidates[candidateIndex][2]:
        print(str(xOutMin[candidateIndex]) + "," + str(yOutMin[candidateIndex]) + "," + str(zOutMin[candidateIndex]) + " = " + str(candidates[candidateIndex][0]) + "," + str(candidates[candidateIndex][1]) + "," + str(candidates[candidateIndex][2]))
    else:
        print("tutto apposto a ferragosto")