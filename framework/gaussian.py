import numpy as np
import math

def get_Gaussian_Kernel(sigma):
    # 3 * 3 Gassian filter
    x, y = np.mgrid[-1:2, -1:2]
    gaussian_kernel = np.exp(-(x**2 + y**2) / (2 * (sigma**2))) * (1 / (2 * math.pi * (sigma**2)))
    # Normalization
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    return gaussian_kernel
