from time import time
from skimage.color import rgb2gray
import numpy as np
from math import log
from skimage.io import imread
from skimage import img_as_float
from blobDetectionMultiprocessing.modules.generateOctave import generateBlurOctave, generateDogOctave
from blobDetectionMultiprocessing.modules.keypoints import filterCandidates, displayKeypoints
from multiprocessing import Pool

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
# candidates = []
startTime = time()


def computeForK(z):
    candidatesList = []
    for i in range(1, dogOctave.shape[0]-1):
        for j in range(1, dogOctave.shape[1]-1):
            miniDog = dogOctave[i-1:i+2, j-1:j+2, z-1:z+2]
            if np.argmax(miniDog) == 13 or np.argmin(miniDog) == 13:
                candidatesList.append([j, i, z])
    return np.array(candidatesList)

p = Pool(dogOctave.shape[2]-2)
# for k in range(1, dogOctave.shape[2]-1):
candidatesLists = p.map(computeForK, range(1, dogOctave.shape[2]-1))

endTime = time()
elapsedTime = endTime - startTime
print("ELAPSED TIME USING MULTIPROCESSING: " + str(round(elapsedTime, 2)) + "s")
npCandidates = np.asarray(candidatesLists)
candidates = []
for candidatesList in candidatesLists:
    for candidate in candidatesList:
        candidates.append(candidate)

print("==========FOUND " + str(len(candidates)) + " CANDIDATES===========")
print("===========FILTERING CANDIDATES===========")

keypoints = filterCandidates(candidates, dogOctave)

print("========REMAINING KEYPOINTS = " + str(len(keypoints)) + "=========")

displayKeypoints(keypoints)



