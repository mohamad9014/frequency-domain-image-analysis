import numpy as np
import cv2
import matplotlib.pyplot as plt

# Create a white image
img = np.ones((512, 512), dtype=np.uint8) * 255

# Apply FFT
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

# Display results
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('White Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('FFT Amplitude Spectrum')
plt.axis('off')
plt.show()

# Load image with line patterns
img_lines = cv2.imread('data/synthetic/lines.png', 0)

f = np.fft.fft2(img_lines)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(img_lines, cmap='gray')
plt.title('Spatial Domain')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('FFT Spectrum')
plt.axis('off')
plt.show()
