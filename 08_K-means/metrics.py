import numpy as np

def euclidean(x1, x2):
    distance = 0
    
    for i in range(len(x1)):
        distance += np.square(x1[i] - x2[i])
    
    return np.sqrt(distance)


def msd(x1, x2):
    distance = 0

    for i in range(len(x1)):
        distance += np.square(x1[i] - x2[i])

    return np.mean(distance)