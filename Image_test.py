# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
#%%

# Load the image
image = cv2.imread('data/camera_01/image_01_1.jpg')  # Replace with your image file path

# Convert the image to RGB (OpenCV uses BGR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Blur the image
image = cv2.GaussianBlur(image, (15, 15), 0)

# Convert the image to grayscale
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Use horizontal Sobel filter
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)

# Use vertical Sobel filter
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

# Combine the two Sobel filters
sobel = np.sqrt(sobel_x**2 + sobel_y**2)

# Plot the original image, the horizontal Sobel filtered image, the vertical Sobel filtered image, and the combined Sobel filtered image
plt.figure(figsize=(20, 20))
plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.subplot(2, 2, 2)
plt.title('Sobel X')
plt.imshow(sobel_x, cmap='gray')
plt.axis('off')
plt.subplot(2, 2, 3)
plt.title('Sobel Y')
plt.imshow(sobel_y, cmap='gray')
plt.axis('off')
plt.subplot(2, 2, 4)
plt.title('Sobel')
plt.imshow(sobel, cmap='gray')
plt.axis('off')
plt.show()




# %%
