import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

# Based on simple totorial on the openCV docs
# https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html

# Load the 2 test images
image_path = 'victoria1.jpg'
image_path2 = 'victoria2.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

# Measure time for ORB function to detect and compute the key points and descripters 
orb = cv2.ORB_create()
start_orb = time.time()
kp_Orb, des_Orb = orb.detectAndCompute(image, None)
kp_Orb2, des_Orb2 = orb.detectAndCompute(image2, None)
end_orb = time.time()
print(f"ORB Keypoint Detection and descripters  Time: {end_orb - start_orb:.4f} seconds")

# Measure time for brute-forcing of orb matches
bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
start_orb_match = time.time()
matches_orb = bf_orb.match(des_Orb, des_Orb2)
matches_orb = sorted(matches_orb, key=lambda x: x.distance)
end_orb_match = time.time()
print(f"ORB Brute-Forcing Matching Time: {end_orb_match - start_orb_match:.4f} seconds")

# Measure time for SIFT function to detect and compute the key points and descripters 
sift = cv2.SIFT_create()
start_sift = time.time()
kp_shift, des_shift = sift.detectAndCompute(image, None)
kp_shift2, des_shift2 = sift.detectAndCompute(image2, None)
end_sift = time.time()
print(f"SIFT Keypoint Detection and descripters Computation Time: {end_sift - start_sift:.4f} seconds")

# Measure time for brute-forcing of sift matches
bf_sift = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
start_sift_match = time.time()
matches_sift = bf_sift.match(des_shift, des_shift2)
matches_sift = sorted(matches_sift, key=lambda x: x.distance)
end_sift_match = time.time()
print(f"SIFT Brute-Forcing Matching Time: {end_sift_match - start_sift_match:.4f} seconds")

# Display best SIFT matches 
matched_image_sift = cv2.drawMatches(image, kp_shift, image2, kp_shift2, matches_sift[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(matched_image_sift)
plt.title("SIFT Matches")
plt.axis('off')
plt.show()

# Display best ORB matches 
matched_image_orb = cv2.drawMatches(image, kp_Orb, image2, kp_Orb2, matches_orb[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(matched_image_orb)
plt.title("ORB Matches")
plt.axis('off')
plt.show()
