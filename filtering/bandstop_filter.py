import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('data/medical/fundus_image.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# FFT
f = np.fft.fft2(gray)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

# Display FFT spectrum
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('FFT Spectrum')
plt.axis('off')
plt.show()

# Create band-stop mask
rows, cols = gray.shape
crow, ccol = rows // 2, cols // 2
mask = np.ones((rows, cols), np.uint8)

# Remove symmetric noise frequencies
cv2.circle(mask, (ccol + 40, crow), 10, 0, -1)
cv2.circle(mask, (ccol - 40, crow), 10, 0, -1)

# Apply mask
f_filtered = fshift * mask
img_filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(f_filtered)))

# Display result
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_filtered, cmap='gray')
plt.title('Band-stop Filtered')
plt.axis('off')
plt.show()
