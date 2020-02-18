from time import time
from skimage.color import rgb2gray
import numpy as np
from skimage.io import imread
from skimage import img_as_float
from modules.generateOctave import generateBlurOctave, generateDogOctave
from modules.keypoints import filterCandidates, displayKeypoints
from modules.partialFunctions import computeForX, computeForY, computeForZ
from multiprocessing import Pool
from functools import partial
import multiprocessing as mp

print("==============LOADING IMAGE===============")

imgPath = "/Volumes/WD/PycharmProjects/BlobDetector/BlobDetection/images/old/1.png"
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

p = Pool(dogOctave.shape[2]-2)
# candidatesLists = p.map(computeForX, range(1, dogOctave.shape[1]-1)) #10.0 secondi
# candidatesLists = p.map(computeForY, range(1, dogOctave.shape[0]-1)) #8.3 secondi
partialComputerForZ = partial(computeForZ, DoGOctave=dogOctave)
candidatesLists = p.map(partialComputerForZ, range(1, dogOctave.shape[2]-1)) #7.45 secondi

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
p.close()
displayKeypoints(keypoints, imgPath)




