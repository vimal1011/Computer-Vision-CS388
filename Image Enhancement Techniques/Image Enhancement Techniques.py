import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read Image and Check if Valid
image_name = (r"C:\Users\hp\Downloads\input image.jpg") # Change this to your actual image file
img = cv2.imread(image_name)

if img is None:
    print("Error: Unable to load image. Please check the file name and format.")
else:
    # 1. Color Space Conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 2. Histogram Equalization (on Grayscale)
    hist_eq = cv2.equalizeHist(gray)

    # 3. Smoothing Filters
    gaussian_blur = cv2.GaussianBlur(img, (5, 5), 0)
    median_blur = cv2.medianBlur(img, 5)

    # 4. Gradient Computation
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Convert gradient images to 8-bit unsigned integers for display
    sobelx_disp = cv2.convertScaleAbs(sobelx)
    sobely_disp = cv2.convertScaleAbs(sobely)
    laplacian_disp = cv2.convertScaleAbs(laplacian)

    # 5. Edge Detection using Canny
    edges = cv2.Canny(gray, 100, 200)

    # --- Display Results ---
    titles = ['Original', 'Grayscale', 'HSV', 'Histogram Equalized',
              'Gaussian Blur', 'Median Blur', 'Sobel X', 'Sobel Y',
              'Laplacian', 'Canny Edge Map']
    
    images = [img, gray, hsv, hist_eq, gaussian_blur, median_blur,
              sobelx_disp, sobely_disp, laplacian_disp, edges]

    plt.figure(figsize=(15, 10))
    for i in range(len(titles)):
        plt.subplot(2, 5, i + 1)
        if titles[i] in ['HSV', 'Original', 'Gaussian Blur', 'Median Blur']:
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()
