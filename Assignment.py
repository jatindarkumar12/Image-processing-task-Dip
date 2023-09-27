import cv2 as cv
import numpy as np

img = cv.imread('bird.jfif')
# cv.imshow('Bird', img )

# ----------------------------   TASK 1   --------------------------

# CHANGING RESOLUTION TO 256x256 PIXELS
resized_img = cv.resize(img, (256, 256))
cv.imshow('Resized Bird', resized_img)

#  ---------------------------   TASK 2   ---------------------------

# RGB TO GRAYSCALE IMAGE
grayscale_img = cv.cvtColor(resized_img, cv.COLOR_RGB2GRAY)
cv.imshow('Grayscale Image', grayscale_img)

#  ---------------------------   TASK 3   ---------------------------

# RGB TO BINARY IMAGE
if resized_img is not None:
    # Define lower and upper bounds for white color (RGB)
    lower_bound = np.array([25,25, 25])  # Lower bound for white color
    upper_bound = np.array([255, 255, 255])  # Upper bound for white color

    # Create a binary mask by thresholding based on the white color range
    binary_mask = cv.inRange(resized_img, lower_bound, upper_bound)


    cv.imshow('Original', resized_img)
    cv.imshow('RGB to Binary', binary_mask)

    cv.waitKey(0)
    cv.destroyAllWindows()
else:
    print("Failed to load the image.")

# GRAYSCALE TO BINARY IMAGE
ret, binary_img2 = cv.threshold(grayscale_img, 127, 255, cv.THRESH_BINARY)
cv.imshow('Grayscale To Binary', binary_img2)

#  ---------------------------   TASK 4   ---------------------------

# CONTOURS
coin_img = cv.imread('Coins.PNG')

'''
USING GAUSSIAN BLUR TO MAKE THE IMAGE BLURRED - THERE EXISTS SOME SMALL GAPS BETWEEN THE
COINS. THE CONTOUR FUNCTION ALSO TENDS TO MAKE CONTOURS AROUND THOSE GAPS. THIS FUNCTION
GAUSSIAN BLUR REMOVES THE CONTOURS MADE FOR SPACES BETWEEN THE COINS. 
'''
blurred_image = cv.GaussianBlur(coin_img, (25, 25), 0)

# CONVERTING IMAGE TO GRAYSCALE
gray = cv.cvtColor(blurred_image, cv.COLOR_BGR2GRAY)

# APPLYING THRESHOLDING
ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# COPYING IMAGE TO MAKE CONTOURS
result_image = coin_img.copy()

# DRAWING CONTOURS
cv.drawContours(result_image, contours, -1, (0, 255, 0), 3)

# DEFINING MIN AND MAX AREA TO IDENTIFY COINS IN THE IMAGE
min_area = 500
max_area = 100000

# APPLYING THE AREA FILTER
filtered_contours = [contour for contour in contours if min_area < cv.contourArea(contour) < max_area]

# PRINTING NO. OF COINS
num_of_coins = len(filtered_contours)
print("Number of coins:", num_of_coins)

# DISPLAYING CONTOURED IMAGE
cv.imshow('Contoured Image', result_image)

cv.waitKey(0)
cv.destroyAllWindows()
