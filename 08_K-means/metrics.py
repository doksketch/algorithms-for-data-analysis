import numpy as np

def get_distance(x1, x2, metrics):
    distance = 0

    for i in range(len(x1)):
        distance += np.square(x1[i] - x2[i])
        if metrics == 'euclidean':
            return np.sqrt(distance)
        else:
            raise ValueError(
                "Incorrect metrics value. Please use: 'euclidean'")

def calc_msd(centroids, clusters):
    msd = 0

    for i in range(len(centroids)):
        msd += np.square(np.sum(centroids[i] - clusters[i]))
    
    return msd / len(clusters)