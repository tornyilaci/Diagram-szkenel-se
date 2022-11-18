# Python program to explain cv2.erode() method

# importing cv2
import cv2

# importing numpy
import numpy as np

# path
path = r'C:\Users\shape.jpg'

# Reading an image in default mode
image = cv2.imread(path)

# Window name in which image is displayed
window_name = 'image'

# Creating kernel
kernel = np.ones((5, 5), np.uint8)

# Using cv2.erode() method
image = cv2.erode(image, kernel)

# Displaying the image
cv2.imshow(window_name, image)