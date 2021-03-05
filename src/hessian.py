"""
File: hessian.py
Author: Shaurya Chandhoke
Description: Helper file which contains the function used for feature extraction via the Hessian detector
"""
from math import pow
from src.image_output_processing import logger

import cv2
import numpy as np


def hessian_suppression(hessian_matrix):
    """
    A helper function that will perform non-maximum suppression on the Hessian matrix.

    :param hessian_matrix: The thresholded matrix
    :return: The Hessian matrix suppressed
    """
    for i in range(len(hessian_matrix) - 3):
        for j in range(len(hessian_matrix[0]) - 3):
            window = hessian_matrix[i:i + 3, j:j + 3]

            # No need to process window of all 0 pixels
            if not np.all((window == 0)):
                maxima = np.max(window)
                window = np.where(window == maxima, 255, 0)
                hessian_matrix[i:i + 3, j:j + 3] = window

    return hessian_matrix


def hessian_determinant(I_xx, I_yy, I_xy, threshold):
    """
    A helper function that will obtain the determinant of the Hessian matrix. Thresholding is performed here.

    :param I_xx: The second pass partial derivative of the image in x after the initial x
    :param I_yy: The second pass partial derivate of the image in y after the initial y
    :param I_xy: The second pass partial derivate of the image in y after the initial x
    :param threshold: The upper bound percentage representing the proportion of points to threshold
    :return: The thresholded Hessian matrix
    """
    hessian_matrix = np.empty(shape=I_xx.shape)

    iterLen_I = len(I_xx)
    iterLen_J = len(I_xx[0])

    # Calculate pixel wise determinant
    for i in range(iterLen_I):
        for j in range(iterLen_J):
            hessian_matrix[i, j] = (I_xx[i, j] * I_yy[i, j]) - pow(I_xy[i, j], 2)

    # Filter by user threshold percentage
    hessian_threshold = np.max(hessian_matrix) * threshold
    hessian_matrix = np.where(hessian_matrix < hessian_threshold, 0, hessian_matrix)

    return hessian_matrix


def hessian_process(inputImage, threshold):
    """
    Main entry point into the feature extraction via the Hessian detector. This method will perform corner detection
    using the Hessian matrix via the Sobel filters as derivative operators. After the features are extracted,
    they are thresholded and then non maximum suppression is applied to reduce to local maxima corners.

    :param inputImage: The image represented as a 2D numpy array. It must have been blurred prior to this.
    :param threshold: The upper bound percentage representing the proportion of points to threshold.
    :return: The detected features of the image.
    """
    logger("1b", 0, "Hessian Feature Detection", substep=True)

    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    sobel_y = np.transpose(sobel_x)

    '''
    Important Disclaimer:

    Using the cv2.filter2D filter function provided by OpenCV due to the time complexity issues that had occurred with
    the original naive implementation. The deprecated implementation had a time complexity of O(n^2).

    This meant that with a sigma value of 5 and a kernel of 31x31, there were ~961 operations that had to be done for
    each pixel, severely growing as the sigma, kernel, or image size increased in size and value.
    '''
    I_x = cv2.filter2D(src=inputImage, ddepth=-1, kernel=sobel_x, borderType=cv2.BORDER_REPLICATE)
    I_y = cv2.filter2D(src=inputImage, ddepth=-1, kernel=sobel_y, borderType=cv2.BORDER_REPLICATE)

    # Second pass of sobel filters for second moment matrices
    I_xx = cv2.filter2D(src=I_x, ddepth=-1, kernel=sobel_x, borderType=cv2.BORDER_REPLICATE)
    I_yy = cv2.filter2D(src=I_y, ddepth=-1, kernel=sobel_y, borderType=cv2.BORDER_REPLICATE)
    I_xy = cv2.filter2D(src=I_x, ddepth=-1, kernel=sobel_y, borderType=cv2.BORDER_REPLICATE)

    hessian_matrix = hessian_determinant(I_xx, I_yy, I_xy, threshold)
    hessian_matrix = hessian_suppression(hessian_matrix)

    logger("1b", 1, "Hessian Feature Detection", substep=True)

    return hessian_matrix
