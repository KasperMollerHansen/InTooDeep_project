# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
#%%

# Load the image
image = cv2.imread('data/camera_01/image_01_1.jpg')  # Replace with your image file path

# Convert the image to HSV
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Split the channels
h, s, v = cv2.split(image_hsv)

# Display all the channels
plt.figure(figsize=(20, 10))

plt.subplot(1, 3, 1)
plt.title("Hue channel")
plt.imshow(h, cmap='gray')
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Saturation channel")
plt.imshow(s, cmap='gray')
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Value channel")
plt.imshow(v, cmap='gray')
plt.axis("off")

plt.show()

# Do the same for the original image (RGB) channels
r, g, b = cv2.split(image)

plt.figure(figsize=(20, 10))

plt.subplot(1, 3, 1)
plt.title("Red channel")
plt.imshow(r, cmap='gray')
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Green channel")
plt.imshow(g, cmap='gray')
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Blue channel")
plt.imshow(b, cmap='gray')
plt.axis("off")

plt.show()



# %%
