import cv2
import matplotlib.pyplot as plt
import numpy as np

# Based on simple totorial on the openCV docs
# https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html

# Load the images
image_path = 'victoria1.jpg'
image_path2 = 'victoria2.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

# Create orb object
orb = cv2.ORB_create()

# Calculate key points
kp = orb.detect(image,None)
kp2 = orb.detect(image2,None)

# with key points calculate descriptors
kp, des = orb.compute(image, kp)
kp2, des2 = orb.compute(image2, kp2)

img = cv2.drawKeypoints(image, kp, None, color=(0,255,0), flags=0)
plt.imshow(img), plt.show()

img2 = cv2.drawKeypoints(image2, kp2, None, color=(0,255,0), flags=0)
plt.imshow(img2), plt.show()