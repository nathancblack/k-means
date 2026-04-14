import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.pyplot as plt
from k_means_nonop import k_means as k_means_nonop
from k_means_op import k_means as k_means_op
import glob
import os
import time

image_dir = os.path.join(os.path.dirname(__file__), "images")
image_paths = sorted(glob.glob(os.path.join(image_dir, "*")))

k = 5

fig, axes = plt.subplots(len(image_paths), 4, figsize=(20, 5 * len(image_paths)))

for row, path in enumerate(image_paths):
    img = Image.open(path)
    data = np.array(img)
    h, w, c = data.shape
    pixels = data.reshape(-1, c)
    name = os.path.basename(path)

    # Scikit-learn
    start = time.time()
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    sklearn_result = kmeans.cluster_centers_[labels].reshape(h, w, c).astype(np.uint8)
    sklearn_time = time.time() - start

    # Custom non-optimized
    start = time.time()
    nonop_result = k_means_nonop(k, img)
    nonop_time = time.time() - start

    # Custom optimized
    start = time.time()
    op_result = k_means_op(k, img)
    op_time = time.time() - start

    print(f"{name}:")
    print(f"  Scikit-learn:    {sklearn_time:.3f}s")
    print(f"  Non-optimized:   {nonop_time:.3f}s")
    print(f"  Optimized:       {op_time:.3f}s")
    print()

    axes[row, 0].imshow(data)
    axes[row, 0].set_title(name)
    axes[row, 1].imshow(sklearn_result)
    axes[row, 1].set_title(f"Scikit-learn ({sklearn_time:.2f}s)")
    axes[row, 2].imshow(nonop_result)
    axes[row, 2].set_title(f"Non-optimized ({nonop_time:.2f}s)")
    axes[row, 3].imshow(op_result)
    axes[row, 3].set_title(f"Optimized ({op_time:.2f}s)")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()
plt.show()
