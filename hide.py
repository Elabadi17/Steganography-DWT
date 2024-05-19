import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt

def cacher_image(cover_image_path, secret_image_path):
    cover_image = cv2.imread(cover_image_path, cv2.IMREAD_GRAYSCALE)
    secret_image = cv2.imread(secret_image_path, cv2.IMREAD_GRAYSCALE)
    
    secret_image = secret_image.astype(float) / 255.0
    
    coeffs = pywt.wavedec2(cover_image, 'haar', level=2)
    cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs
    
    cH1[:secret_image.shape[0], :secret_image.shape[1]] = secret_image
    
    c2 = [cA2, (cH2, cV2, cD2), (cH1, cV1, cD1)]
    stego_image = pywt.waverec2(c2, 'haar')
    
    return stego_image

def extraire_image(stego_image):
    coeffs_stego = pywt.wavedec2(stego_image, 'haar', level=2)
    _, (_, _, _), (cH1_stego, _, _) = coeffs_stego
    
    hidden_image = cH1_stego
    
    return hidden_image

stego_image = cacher_image("image_hote.png", "image_secrete.png")

hidden_image = extraire_image(stego_image)

im = cv2.imread("image_hote.png", cv2.IMREAD_GRAYSCALE)
im_stego = cv2.imread("image_secrete.png", cv2.IMREAD_GRAYSCALE)

im_stego = im_stego.astype(float) / 255.0

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(im, cmap='gray')
plt.title("Cover Image")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(im_stego, cmap='gray')
plt.title("Secret Image")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(stego_image, cmap='gray')
plt.title("Stegano Image")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(hidden_image, cmap='gray')
plt.title("Recovered Image")
plt.axis('off')
plt.tight_layout()

plt.show()
