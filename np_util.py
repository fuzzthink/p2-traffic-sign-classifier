import cv2
import numpy as np


def grayscale(X):
    l = []
    for i in range(len(X)):
        l.append(cv2.cvtColor(X[i], cv2.COLOR_BGR2GRAY))
    gray = np.asarray(l)
    gray.shape += (1,)
    return gray

normalize = lambda x: x / 255. - 0.5
