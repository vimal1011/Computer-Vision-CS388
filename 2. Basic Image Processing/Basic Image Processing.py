import cv2
import numpy as np

# Load the image
image = cv2.imread(r"C:\Users\ELCOT\Desktop\Vimal\input image.jpg")

# Check if image loaded
if image is None:
    print("❌ Image not found or path is incorrect!")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Crop the image
cropped = image[50:200, 100:300]

# Resize the image
resized = cv2.resize(image, (300, 200))

# Apply thresholding
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Convert thresholded to BGR for stacking
thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_img = image.copy()
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

# Blob detection
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 150

# Compatibility for OpenCV 3 and 4
ver = cv2.__version__.split(".")
if int(ver[0]) < 4:
    detector = cv2.SimpleBlobDetector(params)
else:
    detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(gray)
blob_img = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255),
                             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Resize all images to same size and convert to 3-channel if needed
def resize_for_stack(img):
    img_resized = cv2.resize(img, (200, 200))
    if len(img_resized.shape) == 2:  # grayscale to BGR
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
    return img_resized

output = np.hstack([
    resize_for_stack(cropped),
    resize_for_stack(resized),
    resize_for_stack(thresh_bgr),
    resize_for_stack(contour_img),
    resize_for_stack(blob_img)
])

cv2.imshow("Basic Image Processing Results", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
