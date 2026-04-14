import numpy as np
import random
import math
import collections 


# IV: the update step sets each centroid to the mean of its assigned pixels

# Pick trial points: pick k random pixels (initial centroids)

# (1) Set every pixels closest centroid to be the centroid that is closest

# (2) Update centroids to be the mean of its corresponding pixels

# (3) Repeat from 2

class Pixel:
    centroid = 0
    def __init__(self, rgb):
        self.r = int(rgb[0])
        self.g = int(rgb[1])
        self.b = int(rgb[2])

    def __repr__(self):
       return f"(r={(self.r)}, g={self.g}, b={self.b})" 

    def __hash__(self):
        return hash((self.r, self.g, self.b))
    def __eq__(self, other):
        return self.r == other.r and self.g == other.g and self.b == other.b


def rand_pixels(k, pixels, c):
    initial_points = []
    
    while len(initial_points) < k:
        pixel = pixels[random.randint(0, len(pixels) - 1)]
        initial_points.append(pixel)

    return initial_points





def nearest_centroids(centroids, pixels):
      for pixel in pixels:
          closest = 100000
          for centroid in centroids:
              distance = math.sqrt((pixel.r - centroid.r) ** 2 + (pixel.g - centroid.g) ** 2 + (pixel.b - centroid.b) ** 2)
              if distance < closest:
                  closest = distance
                  pixel.centroid = centroid

def k_means(k, image):
    data = np.array(image)
    h, w, c = data.shape
    pixels = data.reshape(-1, c)
    
    # Convert all pixels to Pixel objects
    pixel_objs = []
    for point in pixels:
        pixel = Pixel(point)
        pixel_objs.append(pixel)

    new_centroids = []
    centroids = rand_pixels(k, pixel_objs, c)
    while not set(new_centroids) == set(centroids):
        # Set centroid for all pixels
        nearest_centroids(centroids, pixel_objs)
    
        # Group pixels by centroid
        centroid_groups = collections.defaultdict(list)
        for point in pixel_objs:
            centroid_groups[point.centroid].append(point)

        # Find mean pixel of each centroid group
        new_centroids = []
        for centroid, points in centroid_groups.items():
            mean_r = sum(p.r for p in points) // len(points) 
            mean_g = sum(p.g for p in points) // len(points) 
            mean_b = sum(p.b for p in points) // len(points) 
            new_centroids.append(Pixel([mean_r, mean_g, mean_b]))

        # Update centroids
        centroids = new_centroids
     

    result = np.zeros((h, w, c), dtype=np.uint8)
    for i, point in enumerate(pixel_objs):
        row = i // w
        col = i % w
        result[row, col] = [point.centroid.r, point.centroid.g, point.centroid.b]

    return result

