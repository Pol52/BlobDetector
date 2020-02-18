from pycuda.compiler import SourceModule
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np


# Pycuda section
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

__inline__ __device__ float warpReduceMin(float val){
    for (int offset = warpSize; offset > 0; offset/=2){
        float tmpVal = __shfl_down(val, offset);
        if (tmpVal < val && tmpVal != 0){
            val = tmpVal;
        }
    }
    return val;
}

//Compute max and min for block and save coordinates if max or min is in expected pixel(block center pixel). 
//Can compute together because the block will have either max extrema or min extrema in center pixel
__global__ void findCandidates(float* xOut, float* yOut, float* zOut, float *in, int imageWidth, int imageHeight) {
    float minVal = 999;
    float maxVal = 0;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    
    int pixelIndex = (blockIdx.z + tz) * imageWidth * imageHeight + imageWidth * (blockIdx.y + ty) + blockIdx.x + tx ;
    
    int pixelAmount = (gridDim.z + 2) * imageWidth * imageHeight;    
    if(pixelIndex < pixelAmount){
    
        //Compute block min value
        minVal = min(minVal, in[pixelIndex]);
        __syncthreads();
        minVal = warpReduceMin(minVal);        
        __syncthreads();
        
        //Compute block max value
        maxVal = max(maxVal, in[pixelIndex]);
        __syncthreads();
        maxVal = warpReduceMax(maxVal);
        __syncthreads();
        
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
        
            //Verify min or max value is in expected pixel
            float expectedMin = in[pixelIndex + imageWidth*imageHeight + imageWidth + 1];
            float expectedMax = in[pixelIndex + imageWidth*imageHeight + imageWidth + 1];
            
            if(expectedMin == minVal || expectedMax == maxVal){
                xOut[pixelIndex + imageWidth + 1] = blockIdx.x + 1;
                yOut[pixelIndex + imageWidth + 1] = blockIdx.y + 1;
                zOut[pixelIndex + imageWidth + 1] = blockIdx.z + 1;
            }
        }
    }
}
""")


def computeKeypoints(dogOctave):
    newDogOctave = dogOctave.transpose(2, 0, 1)  # transpose components to obtain (z, y, x)
    xOutMin, yOutMin, zOutMin = findCandidates(newDogOctave)
    candidates = []
    compressCoordinates(xOutMin, yOutMin, zOutMin, candidates)
    return candidates


# create grid depending on DoG octave size and block as 3x3x3 cube containing one possible keypoint candidate
def findCandidates(dogOctave):
    tileWidth = 3
    gridSize = (dogOctave.shape[2] - 2, dogOctave.shape[1] - 2, dogOctave.shape[0] - 2)
    blockSize = (tileWidth, tileWidth, tileWidth)

    totalPixels = dogOctave.shape[0] * dogOctave.shape[1] * dogOctave.shape[2]
    xOutMin = np.zeros((totalPixels,)).astype(np.float32)
    yOutMin = np.zeros((totalPixels,)).astype(np.float32)
    zOutMin = np.zeros((totalPixels,)).astype(np.float32)
    findMin = mod.get_function("findCandidates")
    inputArray = dogOctave.flatten().astype(np.float32)

    findMin(cuda.Out(xOutMin), cuda.Out(yOutMin), cuda.Out(zOutMin),
            cuda.In(inputArray), np.int32(dogOctave.shape[2]), np.int32(dogOctave.shape[1]),
            block=blockSize, grid=gridSize)

    xOutMin = xOutMin.astype(np.int32)
    yOutMin = yOutMin.astype(np.int32)
    zOutMin = zOutMin.astype(np.int32)
    return xOutMin, yOutMin, zOutMin


# Compress coordinates vectors to candidates list removing zeros left after initialization
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

