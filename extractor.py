#%%
import cv2
from matplotlib import pyplot as plt
import numpy as np
from pdf2image import convert_from_path

pdf_path = "/Users/dezso/Desktop/Raw_Documents_Data/pdfexample.pdf"
pages = convert_from_path(pdf_path, 500)

image = np.array(pages[0])

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)


contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

signature_images = []

# takes the 3 biggest elements that are supposed to be signatures
for contour in contours:
    (x, y, w, h) = cv2.boundingRect(contour)
    signature_image = image[y:y+h, x:x+w]
    signature_images.append(signature_image)

for i, sig_img in enumerate(signature_images):
    plt.figure()
    plt.imshow(sig_img, cmap='gray')
    plt.title(f"Signature {i+1}")

plt.show()

# %%