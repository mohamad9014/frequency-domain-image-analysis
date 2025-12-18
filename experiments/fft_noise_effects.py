import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.util import random_noise
from skimage import img_as_float

# Load synthetic image
img = cv2.imread('data/synthetic/lines.png', 0)
img_float = img_as_float(img)

# Add Gaussian noise
sigma = 0.15
noisy_img = random_noise(img_float, var=sigma**2)

# FFT of noisy image
f = np.fft.fft2(noisy_img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

# Display results
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(noisy_img, cmap='gray')
plt.title('Image with Gaussian Noise')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('FFT Spectrum with Noise')
plt.axis('off')
plt.show()
