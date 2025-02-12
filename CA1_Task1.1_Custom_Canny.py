import cv2
import numpy as np
import matplotlib.pyplot as plt


def Apply_suppression(magnatude, angle):
    # create a empty matrix to store the local maximums
    M, N = magnatude.shape
    suppressed = np.zeros((M, N), dtype=np.float32)
   
    # go through every pixel and determine if its a local maximum in the
    # direction if its gradient magnatude.  
    # - structure taken from
    # basic before and after code taken from https://fiveko.com/non-maximum-suppression-gradient/
    #{
    for i in range(1, M-1):
        for j in range(1, N-1):
            # first get the direction and then store the values before and after
            # in that direction for testing later
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                before, after = magnatude[i, j-1], magnatude[i, j+1]
            elif (22.5 <= angle[i, j] < 67.5):
                before, after = magnatude[i-1, j+1], magnatude[i+1, j-1]
            elif (67.5 <= angle[i, j] < 112.5):
                before, after = magnatude[i-1, j], magnatude[i+1, j]
            elif (112.5 <= angle[i, j] < 157.5):
                before, after = magnatude[i-1, j-1], magnatude[i+1, j+1]
           
            # test to see if its a local maximum - if so store that max            
            if magnatude[i, j] >= before and magnatude[i, j] >= after:
                suppressed[i, j] = magnatude[i, j]
            else:
                # not max so dont store
                suppressed[i, j] = 0
            #} end of source
    return suppressed
               


def Cal_mag_direction(image):
    # add zero padding to the image
    padded_image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)


    # apply the basic gaussian filter to the image
    gaussianed_image = cv2.GaussianBlur(padded_image, (7, 7), 0)
   
    # get the x_grad throigh Sobel()
    x_Grad = cv2.Sobel(gaussianed_image, cv2.CV_64F, 1, 0, ksize=3)  
    # get the y_Grad throigh Sobel()
    y_Grad = cv2.Sobel(gaussianed_image, cv2.CV_64F, 0, 1, ksize=3)
   
    # Calculate magnatude
    magnatude = np.sqrt(x_Grad**2 + y_Grad**2)
    # get the angle
    angle = np.arctan2(y_Grad, x_Grad) * (180/np.pi)
    normalized_angle = angle % 180
    return magnatude, normalized_angle


def Apply_threshold(supressed_image,low, high):
    # create a empty matrix to hold the high and low values
    M, N = supressed_image.shape
    thresholded_image = np.zeros((M, N), dtype=np.float32)


    # go through the image and if it is higher then high then store in matrix
    # if it isnt higher then check if its between low and high - if it is
    # store the value low
    for i in range(1, M-1):
        for j in range(1, N-1):
            if supressed_image[i,j] >= high:
                # very strong edge detected
                thresholded_image[i,j] = high
            elif (supressed_image[i,j] <= high) & (supressed_image[i,j] >= low):
                # edge within acceptable paramiters
                thresholded_image[i,j] = low
   
    return thresholded_image
 
def Apply_heurtistic(supressed_image,low , high):
    # a example of a heuristic seen on wikipedia
    # https://en.wikipedia.org/wiki/Canny_edge_detector#:~:text=The%20Canny%20edge%20detector%20is,explaining%20why%20the%20technique%20works.
    # "To track the edge connection, blob analysis is applied by
    # looking at a weak edge pixel and its 8-connected neighborhood
    # pixels. As long as there is one strong edge pixel that is
    # involved in the blob, that weak edge point can be identified
    # as one that should be preserved. These weak edge pixels become
    # strong edges that can then cause their neighboring weak edge
    # pixels to be preserved."
   
    # to do this we go through each pixel and if its low
    # then check all of the nabour values to see if they are high  
   
    M, N = supressed_image.shape
   
    for i in range(1, M-1):
        for j in range(1, N-1):
            # go through each pixel and look for low
            if supressed_image[i,j] == low:
                # we check the nabours
                if ((supressed_image[i+1, j-1:j+2] == high).any()) | ((supressed_image[i-1, j-1:j+2] == high).any()) |  ((supressed_image[i, [j-1, j+1]] == high).any()):
                   # a high navour has been found so we make the low value now high
                    supressed_image[i, j] = high
                else:
                     supressed_image[i, j] = 0
   
    return supressed_image
   
def cannyCustom(image):
    low = 20
    high = 255
    # get a matrix of magnatudes for each pixel as well as angels
    magnatude, angle = Cal_mag_direction(image)
   
    # only collect local maximums from the pixels around it
    suppression = Apply_suppression(magnatude, angle)
   
    # apply the high low threshold to only collate high values
    thresholded = Apply_threshold(suppression,high,low)
   
    # use a heristic blobbing algorithm to get final image
    heristiced = Apply_heurtistic(thresholded,high,low)
   
    return heristiced    
   
   
# load the two images and tnen make them greyscale
image1_path = 'victoria1.jpg'  
image2_path = 'victoria2.jpg'  
image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)


# Apply custom Canny edge function
custom_edges1 = cannyCustom(image1)
custom_edges2 = cannyCustom(image2)


# use OpenCVs built in Canny edge detector function
opencv_edges1 = cv2.Canny(image1, 50, 150)
opencv_edges2 = cv2.Canny(image2, 50, 150)



# set up display system
#{ display code taken from this totorial on matplotlib.pyplot
# https://scikit-image.org/docs/stable/auto_examples/applications/plot_image_comparison.html

# Display results using plt as i am more comfortable with that library
fig, axs = plt.subplots(2, 3, figsize=(15, 10))


axs[0, 0].imshow(image1, cmap='gray')
axs[0, 0].set_title('Original Image 1')
axs[0, 1].imshow(custom_edges1, cmap='gray')
axs[0, 1].set_title('Custom Canny Image 1')
axs[0, 2].imshow(opencv_edges1, cmap='gray')
axs[0, 2].set_title('OpenCV Canny Image 1')

axs[1, 0].imshow(image2, cmap='gray')
axs[1, 0].set_title('Original Image 2')
axs[1, 1].imshow(custom_edges2, cmap='gray')
axs[1, 1].set_title('Custom Canny Image 2')
axs[1, 2].imshow(opencv_edges2, cmap='gray')
axs[1, 2].set_title('OpenCV Canny Image 2')


# Hide unused axes
for ax in axs.flat:
    ax.axis('off')


plt.tight_layout()
plt.show()
# }
