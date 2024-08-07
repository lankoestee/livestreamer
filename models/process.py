import cv2
import numpy as np

def invert(image):
    return cv2.bitwise_not(image)

def sobel(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)
    sobel = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
    return sobel

def roberts(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    roberts_x = np.array([[1, 0], [0, -1]])
    roberts_y = np.array([[0, 1], [-1, 0]])
    roberts_x = cv2.filter2D(gray, cv2.CV_64F, roberts_x)
    roberts_y = cv2.filter2D(gray, cv2.CV_64F, roberts_y)
    abs_roberts_x = cv2.convertScaleAbs(roberts_x)
    abs_roberts_y = cv2.convertScaleAbs(roberts_y)
    roberts = cv2.addWeighted(abs_roberts_x, 0.5, abs_roberts_y, 0.5, 0)
    return roberts

def laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    abs_laplacian = cv2.convertScaleAbs(laplacian)
    return abs_laplacian