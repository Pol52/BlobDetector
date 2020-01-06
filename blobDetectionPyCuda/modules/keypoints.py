import math
import cv2
import numpy as np
import numpy.linalg as linalg


def filterCandidates(candidates, dogOctave):
    keypoints = []
    r_th = 10
    t_c = 0.03
    R_th = (r_th+1)**2 / r_th

    for candidate in candidates:
        y, x, s = candidate[1], candidate[0], candidate[2]
        dx = (dogOctave[y, x+1, s]-dogOctave[y, x-1, s])/2.
        dy = (dogOctave[y+1, x, s]-dogOctave[y-1, x, s])/2.
        ds = (dogOctave[y, x, s+1]-dogOctave[y, x, s-1])/2.

        dxx = dogOctave[y, x+1, s]-2*dogOctave[y, x, s]+dogOctave[y, x-1, s]
        dxy = ((dogOctave[y+1, x+1, s]-dogOctave[y+1, x-1, s]) - (dogOctave[y-1, x+1, s]-dogOctave[y-1, x-1, s]))/4.
        dxs = ((dogOctave[y, x+1, s+1]-dogOctave[y, x-1, s+1]) - (dogOctave[y, x+1, s-1]-dogOctave[y, x-1, s-1]))/4.
        dyy = dogOctave[y+1, x, s]-2*dogOctave[y, x, s]+dogOctave[y-1, x, s]
        dys = ((dogOctave[y+1, x, s+1]-dogOctave[y-1, x, s+1]) - (dogOctave[y+1, x, s-1]-dogOctave[y-1, x, s-1]))/4.
        dss = dogOctave[y, x, s+1]-2*dogOctave[y, x, s]+dogOctave[y, x, s-1]

        J = np.array([dx, dy, ds])
        HD = np.array([
            [dxx, dxy, dxs],
            [dxy, dyy, dys],
            [dxs, dys, dss]])

        offset = -linalg.inv(HD).dot(J)
        contrast = dogOctave[y, x, s] + .5*J.dot(offset)
        if abs(contrast) < t_c:
            continue

        w, v = linalg.eig(HD)
        r = w[1]/w[0]
        R = (r+1)**2 / r
        if R > R_th:
            continue

        kp = np.array([x, y, s]) + offset
        if kp[1] >= dogOctave.shape[0] or kp[0] >= dogOctave.shape[1]:
            continue

        keypoints.append(kp)
    return keypoints


def displayKeypoints(keypoints):
    resultImage = cv2.imread('/WD/PycharmProjects/BlobDetector/blobDetectionSequential/images/1.png', cv2.COLOR_RGB2BGR)
    for keypoint in keypoints:#keypoint sono x,y,candidates sono y,x
        scalespace = int(np.ceil(keypoint[2]))
        if scalespace >= 0:
            radius = math.sqrt(2)*scalespace*1.6
            resultImage = cv2.circle(resultImage, (int(keypoint[0]), int(keypoint[1])), int(radius), (0, 255, 0), 2)
    cv2.imshow('Result', resultImage)
    cv2.waitKey(0)
