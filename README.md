Canny Edge Detection (Custom Implementation)

Overview

This project implements a custom version of the Canny Edge Detection algorithm from scratch, focusing on optimization techniques such as gradient assignment and non-maximum suppression. The goal is to explore the core mechanics of edge detection while comparing the results to OpenCV’s built-in Canny edge detector.

The Canny Edge Detection algorithm is widely used in computer vision applications for detecting edges in an image by emphasizing regions with high-intensity changes. Instead of using OpenCV’s implementation, this project re-implements the algorithm step by step to gain a deeper understanding of its functionality and optimization possibilities.

Why Implement a Custom Canny Edge Detector?

1. Deepening Computer Vision Knowledge

Understanding how edge detection works is fundamental to many applications in image processing, such as object detection, segmentation, and feature extraction. By implementing the Canny Edge Detector from scratch, this project demonstrates an in-depth understanding of gradient computation, non-maximum suppression, and edge linking techniques.

2. Optimization and Customization

Many real-world applications require fine-tuned edge detection beyond what OpenCV provides. By implementing a custom solution, adjustments can be made to:

Improve gradient calculation.

Optimize non-maximum suppression.

Experiment with different heuristics for edge linking.

3. Performance Benchmarking

Comparing the performance of the custom implementation to OpenCV’s built-in Canny function helps analyze the trade-offs in speed, accuracy, and computational efficiency.

How Canny Edge Detection Works

The Canny Edge Detection algorithm consists of several key steps:

Step 1: Gaussian Smoothing

Raw images often contain noise, which can lead to false edge detections.

A Gaussian filter is applied to smooth the image and reduce noise before computing gradients.

Implemented using OpenCV’s GaussianBlur() with a kernel size of (7,7).

Step 2: Computing Gradient Magnitude and Direction

The intensity gradients of the image are calculated using the Sobel operator.

Gradients are computed in both x and y directions:

x_grad = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)

y_grad = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

Gradient magnitude is computed as:

magnitude = np.sqrt(x_grad**2 + y_grad**2)

Gradient direction is calculated using np.arctan2(y_grad, x_grad) and normalized to 180 degrees.

Step 3: Non-Maximum Suppression

Only the strongest edge responses are retained.

The gradient direction is used to compare pixel intensities along the edge direction.

If the current pixel’s magnitude is greater than its neighbors along the gradient direction, it is preserved; otherwise, it is suppressed (set to zero).

Step 4: Double Thresholding

Two threshold values (low and high) are applied:

Pixels with a gradient magnitude higher than the high threshold are considered strong edges.

Pixels between the low and high thresholds are weak edges and may be linked to strong edges.

Pixels below the low threshold are suppressed.

Step 5: Edge Tracking by Hysteresis (Heuristic-based Edge Linking)

Weak edges are examined in an 8-connected neighborhood to check if they are connected to a strong edge.

If a weak edge has at least one strong edge neighbor, it is retained; otherwise, it is discarded.

This step ensures that real edges are preserved while removing false detections caused by noise.

Implementation Details

The entire process is implemented in Python using NumPy and OpenCV, ensuring modular and readable code. The key functions include:

calc_mag_direction(image): Computes gradient magnitudes and directions.

apply_suppression(magnitude, angle): Performs non-maximum suppression.

apply_threshold(suppressed_image, low, high): Applies double thresholding.

apply_heuristic(suppressed_image, low, high): Implements edge tracking by hysteresis.

canny_custom(image): Orchestrates all steps to generate the final edge-detected image.

Running the Code

Installation

Ensure you have the required dependencies installed:

pip install opencv-python numpy matplotlib

Running the Script

Place grayscale images in the same directory as the script and update the file names accordingly.

python canny_edge_detection.py

Expected Output

The script will display:

The original images.

Edge-detected images using the custom Canny implementation.

Edge-detected images using OpenCV’s built-in Canny function.

A side-by-side comparison of both methods.

Performance Benchmark

Performance comparisons between the custom implementation and OpenCV’s Canny function are conducted.
Metrics include:

Execution time.

Edge detection accuracy.

Differences in noise sensitivity.

Example Output

A set of images is displayed showing:

Custom Canny Edge Detection output.

OpenCV’s Canny Edge Detection output.

Visual comparisons between the two methods.

Future Enhancements

Potential improvements include:

Adaptive Thresholding: Automatically selecting the best threshold values.

GPU Acceleration: Implementing the algorithm using CUDA for faster processing.

Optimization using Vectorized Operations: Reducing execution time with NumPy’s optimized operations.

Integrating Machine Learning Approaches: Using deep learning to refine edge detection.

References

OpenCV Documentation: https://docs.opencv.org/

Canny Edge Detector Wiki: https://en.wikipedia.org/wiki/Canny_edge_detector

Scikit-Image Tutorials: https://scikit-image.org/

License

MIT License

