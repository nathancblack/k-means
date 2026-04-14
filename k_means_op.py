import numpy as np
import random
import math
import collections 

def k_means(k, image):
    data = np.array(image)
    h, w, c = data.shape
    pixels = data.reshape(-1, c).astype(np.float64) 
    


    indices = np.random.choice(len(pixels), size=k, replace=False)
    centroids = pixels[indices]

    while True:
        distances = np.linalg.norm(pixels[:, np.newaxis] - centroids[np.newaxis, :], axis=2)
        labels = np.argmin(distances, axis=1)  # closest centroid index for each pixel

        new_centroids = np.array([pixels[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i] for i in range(k)])
        if np.array_equal(new_centroids, centroids):
            break
        centroids = new_centroids

    return centroids[labels].reshape(h, w, c).astype(np.uint8)