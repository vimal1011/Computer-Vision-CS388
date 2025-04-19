import cv2
import numpy as np

# Load the image
image = cv2.imread(r'C:\Users\ELCOT\Desktop\Vimal\Image rotation\input image.jpg')

if image is None:
    print("Error: Could not load image. Check the path!")
    exit()

# Get the image dimensions (height and width)
(h, w) = image.shape[:2]

# Set the center of rotation
center = (w // 2, h // 2)

# Set the rotation angle (in degrees) and scale (1.0 = no scaling)
angle = 45
scale = 1.0

# Get the rotation matrix
M = cv2.getRotationMatrix2D(center, angle, scale)

# Perform the rotation
rotated = cv2.warpAffine(image, M, (w, h))

# Show the result
cv2.imshow("Rotated Image", rotated)
cv2.waitKey(0)
cv2.destroyAllWindows(2)
