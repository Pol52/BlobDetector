#!/usr/bin/env python
from time import time
from skimage.color import rgb2gray
import numpy as np
from math import log
from skimage.io import imread
from skimage import img_as_float
import cv2
import math
from blobDetectionPyCuda.modules.generateOctave import generateBlurOctave, generateDogOctave
from blobDetectionPyCuda.modules.pycuda import computeKeypoints
from blobDetectionPyCuda.modules.keypoints import filterCandidates, displayKeypoints


print("==============LOADING IMAGE===============")

original = imread('/WD/PycharmProjects/BlobDetector/blobDetectionSequential/images/1.png')
grayscaleImage = rgb2gray(original)
image = img_as_float(grayscaleImage)

maxSigma = 100
minSigma = 0.1
sigmaRatio = 1.4
k = int(log(float(maxSigma) / minSigma, sigmaRatio)) + 1
sigmaList = np.array([minSigma + (sigmaRatio ** i) for i in range(k + 1)])
print("===============IMAGE LOADED===============")
print("=========GENERATING BLURRED IMAGES========")

octave = generateBlurOctave(sigmaList, image)

print("=====BUILDING DIFFERENCE OF GAUSSIANS=====")

dogOctave = generateDogOctave(octave)

print("==========LOOKING FOR CANDIDATES==========")

startTime = time()
candidates = computeKeypoints(dogOctave)
endTime = time()
elapsedTime = endTime - startTime
print("=====ELAPSED TIME USING PYCUDA: " + str(round(elapsedTime, 2)) + "s=====")

print("==========FOUND " + str(len(candidates)) + " CANDIDATES===========")
print("===========FILTERING CANDIDATES===========")

keypoints = filterCandidates(candidates, dogOctave)

print("========REMAINING KEYPOINTS = " + str(len(keypoints)) + "=========")

displayKeypoints(keypoints)

