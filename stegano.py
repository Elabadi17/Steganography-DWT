import numpy as np
import pywt
import matplotlib.pyplot as plt
import cv2
import random


def cacher_image(image, steg):
    coeffs = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs
    coeffs1= pywt.dwt2(HH, 'haar')
    HH_mod = steg
    coeffs_mod = LL, (LH, HL, HH_mod)
    image_stegano = pywt.idwt2(coeffs_mod, 'haar')
    return image_stegano

def extraire_image(image_stegano):
    coeffs = pywt.dwt2(image_stegano, 'haar')
    LL, (LH, HL, HH) = coeffs
    return HH

image_hote = cv2.imread('image_hote.png', cv2.IMREAD_GRAYSCALE)
image_secrete = cv2.imread("image_secrete.png", cv2.IMREAD_GRAYSCALE)

image_stegano = cacher_image(image_hote, image_secrete)

image_extraite = extraire_image(image_stegano)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(image_hote, cmap='gray')
plt.title("Cover Image")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(image_secrete, cmap='gray')
plt.title("Secret Image")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(image_stegano, cmap='gray')
plt.title("Stegano Image")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(image_extraite, cmap='gray')
plt.title("Recovered Image")
plt.axis('off')

plt.tight_layout()
plt.show()
