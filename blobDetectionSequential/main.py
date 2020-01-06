from time import time
from skimage.color import rgb2gray
import numpy as np
from math import log
from skimage.io import imread
from skimage import img_as_float
from blobDetectionSequential.modules.generateOctave import generateBlurOctave, generateDogOctave
from blobDetectionSequential.modules.keypoints import filterCandidates, displayKeypoints

maxSigma = 100
minSigma = 0.1
sigmaRatio = 1.4

print("==============LOADING IMAGE===============")

original = imread('/WD/PycharmProjects/BlobDetector/blobDetectionSequential/images/1.png')
grayscaleImage = rgb2gray(original)
# grayscaleImage = cv2.imread('/WD/PycharmProjects/BlobDetector/blobDetection/images/1.png', cv2.IMREAD_GRAYSCALE)
image = img_as_float(grayscaleImage)
k = int(log(float(maxSigma) / minSigma, sigmaRatio)) + 1
sigmaList = np.array([minSigma + (sigmaRatio ** i) for i in range(k + 1)])

print("===============IMAGE LOADED===============")
print("=========GENERATING BLURRED IMAGES========")

octave = generateBlurOctave(sigmaList, image)

print("=====BUILDING DIFFERENCE OF GAUSSIANS=====")

dogOctave = generateDogOctave(octave)

print("==========LOOKING FOR CANDIDATES==========")
candidates = []
startTime = time()

for i in range(1, dogOctave.shape[0]-1):
    for j in range(1, dogOctave.shape[1]-1):
        for k in range(1, dogOctave.shape[2]-1):
            patch = dogOctave[i-1:i+2, j-1:j+2, k-1:k+2]
            if np.argmax(patch) == 13 or np.argmin(patch) == 13:
                candidates.append([i, j, k])

endTime = time()
elapsedTime = endTime - startTime
print("======ELAPSED TIME USING FOR: " + str(round(elapsedTime, 2)) + "s======")
print("==========FOUND " + str(len(candidates)) + " CANDIDATES===========")
print("===========FILTERING CANDIDATES===========")

keypoints = filterCandidates(candidates, dogOctave)

print("========REMAINING KEYPOINTS = " + str(len(keypoints)) + "=========")

displayKeypoints(keypoints)