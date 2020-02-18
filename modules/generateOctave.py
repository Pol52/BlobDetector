from scipy.ndimage import gaussian_filter
import numpy as np
from math import log


def generateBlurOctave(image):
    maxSigma = 50
    minSigma = 0.1
    sigmaRatio = 1.4
    k = int(log(float(maxSigma) / minSigma, sigmaRatio)) + 1
    sigmaList = np.array([minSigma + (sigmaRatio ** i) for i in range(k + 1)])
    octave = []
    for sigma in sigmaList:
        blurredImage = gaussian_filter(image, sigma)
        octave.append(blurredImage)
    return octave


def generateDogOctave(octave):
    tempDogOctave = []
    for index in range(1, len(octave)):
        dog = octave[index] - octave[index-1]
        tempDogOctave.append(dog)
    dogOctave = np.concatenate([o[:, :, np.newaxis] for o in tempDogOctave], axis=2)
    return dogOctave

