from pycuda.compiler import SourceModule
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np


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

__global__ void findMax(float *xOut, float *yOut, float *zOut, float *in, int imageWidth, int imageHeight) { 
    float maxVal = 0;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;    
    int pixelIndex =  (blockIdx.z + tz) * imageWidth * imageHeight + imageWidth * (blockIdx.y + ty) + blockIdx.x + tx ;
    int pixelAmount = (blockDim.z + 2) * imageWidth * imageHeight;
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
                xOut[pixelIndex + imageWidth + 1] = blockIdx.x + 1;
                yOut[pixelIndex + imageWidth + 1] = blockIdx.y + 1;
                zOut[pixelIndex + imageWidth + 1] = blockIdx.z + 1;
            }
        }        
    }
}

__inline__ __device__ float warpReduceMin(float val){
    for (int offset = warpSize; offset > 0; offset/=2){
        float tmpVal = __shfl_down(val, offset);
        if (tmpVal < val && tmpVal != 0){
            val = tmpVal;
        }
    }
    return val;
}

__global__ void findMin(float* xOut, float* yOut, float* zOut, float *in, int imageWidth, int imageHeight) {
    float minVal = 999;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int pixelIndex = (blockIdx.z + tz) * imageWidth * imageHeight + imageWidth * (blockIdx.y + ty) + blockIdx.x + tx ;

    int pixelAmount = (blockDim.z + 2) * imageWidth * imageHeight;
    if(pixelIndex < pixelAmount){
        minVal = min(minVal, in[pixelIndex]);
        __syncthreads();
        minVal = warpReduceMin(minVal);
        __syncthreads();
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
            float expectedMin = in[pixelIndex + imageWidth*imageHeight + imageWidth + 1];
            if(expectedMin == minVal ){
                xOut[pixelIndex + imageWidth + 1] = blockIdx.x + 1;
                yOut[pixelIndex + imageWidth + 1] = blockIdx.y + 1;
                zOut[pixelIndex + imageWidth + 1] = blockIdx.z + 1;
            }
        }
    }
}
""")


def computeKeypoints(dogOctave):

    newDogOctave = dogOctave.transpose(2, 0, 1)
    xOutMax, yOutMax, zOutMax = findMaxPoints(newDogOctave)
    xOutMin, yOutMin, zOutMin = findMinPoints(newDogOctave)
    candidates = []
    compressCoordinates(xOutMax, yOutMax, zOutMax, candidates)
    compressCoordinates(xOutMin, yOutMin, zOutMin, candidates)

    return candidates


def findMaxPoints(newDogOctave):
    gridSize = (newDogOctave.shape[2] - 2, newDogOctave.shape[1] - 2, newDogOctave.shape[0] - 2)
    tileWidth = 3
    blockSize = (tileWidth, tileWidth, tileWidth)
    totalPixels = newDogOctave.shape[0] * newDogOctave.shape[1] * newDogOctave.shape[2]
    xOutMax = np.zeros((totalPixels,)).astype(np.float32)
    yOutMax = np.zeros((totalPixels,)).astype(np.float32)
    zOutMax = np.zeros((totalPixels,)).astype(np.float32)
    findMax = mod.get_function("findMax")
    inputArray = newDogOctave.flatten().astype(np.float32)
    findMax(cuda.Out(xOutMax), cuda.Out(yOutMax), cuda.Out(zOutMax),
            cuda.In(inputArray), np.int32(newDogOctave.shape[2]), np.int32(newDogOctave.shape[1]),
            block=blockSize, grid=gridSize)
    xOutMax = xOutMax.astype(np.int32)
    yOutMax = yOutMax.astype(np.int32)
    zOutMax = zOutMax.astype(np.int32)
    return xOutMax, yOutMax, zOutMax


def findMinPoints(newDogOctave):
    gridSize = (newDogOctave.shape[2] - 2, newDogOctave.shape[1] - 2, newDogOctave.shape[0] - 2)
    tileWidth = 3
    blockSize = (tileWidth, tileWidth, tileWidth)
    totalPixels = newDogOctave.shape[0] * newDogOctave.shape[1] * newDogOctave.shape[2]
    xOutMin = np.zeros((totalPixels,)).astype(np.float32)
    yOutMin = np.zeros((totalPixels,)).astype(np.float32)
    zOutMin = np.zeros((totalPixels,)).astype(np.float32)
    findMin = mod.get_function("findMin")
    inputArray = newDogOctave.flatten().astype(np.float32)
    findMin(cuda.Out(xOutMin), cuda.Out(yOutMin), cuda.Out(zOutMin),
            cuda.In(inputArray), np.int32(newDogOctave.shape[2]), np.int32(newDogOctave.shape[1]),
            block=blockSize, grid=gridSize)
    xOutMin = xOutMin.astype(np.int32)
    yOutMin = yOutMin.astype(np.int32)
    zOutMin = zOutMin.astype(np.int32)
    return xOutMin, yOutMin, zOutMin


def compressCoordinates(xOut, yOut, zOut, candidates):
    xOut = np.ma.masked_equal(xOut, 0).compressed()
    yOut = np.ma.masked_equal(yOut, 0).compressed()
    zOut = np.ma.masked_equal(zOut, 0).compressed()
    for i in range(0, xOut.shape[0]):
        if xOut[i] != 0:
            valX = xOut[i]
            valY = yOut[i]
            valZ = zOut[i]
            candidates.append([valX, valY, valZ])

