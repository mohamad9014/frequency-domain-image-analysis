import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load medical image
img = cv2.imread('data/medical/fundus_image.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# FFT
f = np.fft.fft2(gray)
fshift = np.fft.fftshift(f)

rows, cols = gray.shape
crow, ccol = rows // 2, cols // 2

# Low-pass filter
mask_lp = np.zeros((rows, cols), np.uint8)
radius = 30
cv2.circle(mask_lp, (ccol, crow), radius, 1, -1)

# High-pass filter
mask_hp = 1 - mask_lp

# Apply low-pass
f_lp = fshift * mask_lp
img_lp = np.abs(np.fft.ifft2(np.fft.ifftshift(f_lp)))

# Apply high-pass
f_hp = fshift * mask_hp
img_hp = np.abs(np.fft.ifft2(np.fft.ifftshift(f_hp)))

# Display results
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(gray, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(img_lp, cmap='gray')
plt.title('Low-pass Filtered')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(img_hp, cmap='gray')
plt.title('High-pass Filtered')
plt.axis('off')

plt.show()
