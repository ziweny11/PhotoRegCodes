#tmp script for calculating k for scaling between gs and dust3r pc
import cv2
import numpy as np
import matplotlib.pyplot as plt


gsdepth = np.load('d1.npy')
gsweight = np.load('w1.npy')
d3depth = np.load('../dust3r/d3depth.npy')
d3conf = np.load('../dust3r/d3confs.npy')

print(gsdepth.shape, gsweight.shape, d3depth.shape, d3conf.shape)


# Create the heatmap using plt.imshow()
plt.figure(figsize=(10, 6))  # Set the figure size as desired
plt.imshow(d3depth, cmap='hot', interpolation='nearest')  # 'hot' colormap for heatmap
plt.colorbar()  # Add a colorbar to show the confidence values
plt.title('Confidence Map Heatmap')
plt.xlabel('Pixel X')
plt.ylabel('Pixel Y')
plt.show()