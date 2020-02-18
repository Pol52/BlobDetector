#!/usr/bin/env python
from time import time
from skimage.color import rgb2gray
import numpy as np
from math import log
from skimage.io import imread
from skimage import img_as_float

from BlobDetection.modules.generateOctave import generateBlurOctave, generateDogOctave
from BlobDetection.modules.pycuda import computeKeypoints
from BlobDetection.modules.keypoints import filterCandidates, displayKeypoints


print("==============LOADING IMAGE===============")

imgPath = "/WD/PycharmProjects/BlobDetector/BlobDetection/images/old/1.png"
original = imread(imgPath)
grayscaleImage = rgb2gray(original)
image = img_as_float(grayscaleImage)

print("===============IMAGE LOADED===============")
print("=========GENERATING BLURRED IMAGES========")

octave = generateBlurOctave(image)

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

displayKeypoints(keypoints, imgPath)

