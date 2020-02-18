import math
import cv2
import numpy as np
import numpy.linalg as linalg


def filterCandidates(candidates, dogOctave):
    keypoints = []
    r_th = 50
    t_c = 0.04
    R_th = (r_th+1)**2 / r_th

    for candidate in candidates:
        x, y, z = candidate[0], candidate[1], candidate[2]
        dx = (dogOctave[y, x+1, z]-dogOctave[y, x-1, z])/2.
        dy = (dogOctave[y+1, x, z]-dogOctave[y-1, x, z])/2.
        ds = (dogOctave[y, x, z+1]-dogOctave[y, x, z-1])/2.

        dxx = dogOctave[y, x+1, z]-2*dogOctave[y, x, z]+dogOctave[y, x-1, z]
        dxy = ((dogOctave[y+1, x+1, z]-dogOctave[y+1, x-1, z]) - (dogOctave[y-1, x+1, z]-dogOctave[y-1, x-1, z]))/4.
        dxs = ((dogOctave[y, x+1, z+1]-dogOctave[y, x-1, z+1]) - (dogOctave[y, x+1, z-1]-dogOctave[y, x-1, z-1]))/4.
        dyy = dogOctave[y+1, x, z]-2*dogOctave[y, x, z]+dogOctave[y-1, x, z]
        dys = ((dogOctave[y+1, x, z+1]-dogOctave[y-1, x, z+1]) - (dogOctave[y+1, x, z-1]-dogOctave[y-1, x, z-1]))/4.
        dss = dogOctave[y, x, z+1]-2*dogOctave[y, x, z]+dogOctave[y, x, z-1]

        # Compute Jacobian and Hessian matrices to obtain the candidate position offset from the previously
        # found coordinates
        J = np.array([dx, dy, ds])
        HD = np.array([
            [dxx, dxy, dxs],
            [dxy, dyy, dys],
            [dxs, dys, dss]])

        try:
            offset = -linalg.inv(HD).dot(J)
        except linalg.LinAlgError:
            continue

        contrast = dogOctave[y, x, z] + .5*J.dot(offset)
        if abs(contrast) < t_c:
            continue

        # Compute eigenvalues to identify and discard edges
        w, v = linalg.eig(HD)
        r = w[1]/w[0]
        R = (r+1)**2 / r
        if R > R_th:
            continue

        kp = np.array([x, y, z]) + offset
        if kp[1] >= dogOctave.shape[0] or kp[0] >= dogOctave.shape[1]:
            continue

        keypoints.append(kp)
    return keypoints


def displayKeypoints(keypoints, imgPath):
    resultImage = cv2.imread(imgPath, cv2.COLOR_RGB2BGR)

    for keypoint in keypoints:#keypoint sono x,y,candidates sono y,x
        if keypoint[0] > 0 and keypoint[1] > 0:
            scalespace = int(np.ceil(keypoint[2]))
            if scalespace >= 0:
                radius = math.sqrt(2)*scalespace*1.4
                resultImage = cv2.circle(resultImage, (int(keypoint[0]), int(keypoint[1])), int(radius), (0, 255, 0), 2)

    cv2.imshow('Result', resultImage)
    cv2.waitKey(0)
