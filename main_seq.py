from time import time
from skimage.color import rgb2gray
import numpy as np
from math import log
from skimage.io import imread
from skimage import img_as_float
from BlobDetection.modules.generateOctave import generateBlurOctave, generateDogOctave
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
candidates = []
startTime = time()

for y in range(1, dogOctave.shape[0] - 1):
    for x in range(1, dogOctave.shape[1] - 1):
        for z in range(1, dogOctave.shape[2] - 1):
            patch = dogOctave[y - 1:y + 2, x - 1:x + 2, z - 1:z + 2]
            if np.argmax(patch) == 13 or np.argmin(patch) == 13:
                candidates.append([x, y, z])

endTime = time()
elapsedTime = endTime - startTime
print("======ELAPSED TIME USING FOR: " + str(round(elapsedTime, 2)) + "s======")
print("==========FOUND " + str(len(candidates)) + " CANDIDATES===========")
print("===========FILTERING CANDIDATES===========")

keypoints = filterCandidates(candidates, dogOctave)

print("========REMAINING KEYPOINTS = " + str(len(keypoints)) + "=========")

displayKeypoints(keypoints, imgPath)
