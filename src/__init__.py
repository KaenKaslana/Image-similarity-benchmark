"""Image similarity benchmark.

Compares pairs of images (reference vs candidate, matched by file name) using
SSIM, LPIPS, silhouette IoU and a Chamfer-style edge similarity, and combines
them into a weighted 0-100 score.
"""

__version__ = "0.1.0"
