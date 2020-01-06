from scipy.ndimage import gaussian_filter
import numpy as np


def generateBlurOctave(sigmaList, image):
    octave = []
    for sigma in sigmaList:
        blurredImage = gaussian_filter(image, sigma)
        # blurredImage = cv2.GaussianBlur(image, (33,33), sigmaX=sigma, borderType = cv2.BORDER_DEFAULT)
        octave.append(blurredImage)
    return octave


def generateDogOctave(octave):
    tempDogOctave = []
    for index in range(1, len(octave)):
        dog = octave[index] - octave[index-1]
        # io.imsave(str(index) + ".jpg", dog)
        tempDogOctave.append(dog)
    dogOctave = np.concatenate([o[:, :, np.newaxis] for o in tempDogOctave], axis=2)
    return dogOctave

