import numpy as np


def computeForZ(z, DoGOctave):
    tempList = []
    for i in range(1, DoGOctave.shape[0]-1):
        for j in range(1, DoGOctave.shape[1]-1):
            miniDog = DoGOctave[i-1:i+2, j-1:j+2, z-1:z+2]
            if np.argmin(miniDog) == 13 or np.argmax(miniDog) == 13:
                tempList.append([j, i, z])
    return np.array(tempList)


def computeForY(y, DoGOctave):
    tempList = []
    for i in range(1, DoGOctave.shape[1]-1):
        for j in range(1, DoGOctave.shape[2]-1):
            miniDog = DoGOctave[y-1:y+2, i-1:i+2, j-1:j+2]
            if np.argmin(miniDog) == 13 or np.argmax(miniDog) == 13:
                tempList.append([i, y, j])
    return np.array(tempList)


def computeForX(x, DoGOctave):
    tempList = []
    for i in range(1, DoGOctave.shape[0]-1):
        for j in range(1, DoGOctave.shape[2]-1):
            miniDog = DoGOctave[i-1:i+2, x-1:x+2, j-1:j+2]
            if np.argmin(miniDog) == 13 or np.argmax(miniDog) == 13:
                tempList.append([x, i, j])
    return np.array(tempList)
