"""
File: hough.py
Author: Shaurya Chandhoke
Description: Helper file which contains the function used for line detection via the Hough Transform method
"""
from math import sin, cos
from src.image_output_processing import logger

import cv2
import numpy as np


def draw_lines(hough_accumulator, feature_matrix, rho_linespace, theta_linespace):
    """
    Helper function that will obtain the (p, theta) points representing the highest tallied locations in the Hough
    space. It will convert each highest point detected back into the Cartesian plane and draw a line in the feature
    matrix.

    :param hough_accumulator: The Hough space matrix
    :param feature_matrix: The Hessian feature matrix
    :param rho_linespace: The p axis of the hough accumulator
    :param theta_linespace: The theta axis of the hough accumulator
    :return: The Hessian feature matrix with lines drawn
    """
    voteTallies = hough_accumulator.flatten()
    voteTallies = np.unique(np.sort(voteTallies))
    voteTallies = voteTallies[::-1]

    x_0 = 0
    x_end = feature_matrix.shape[1]

    i = 0
    while i != 4:
        vote = voteTallies[i]
        row, col = np.where(hough_accumulator == vote)

        for p_index, theta_index in zip(row, col):
            p = rho_linespace[p_index]
            theta = theta_linespace[theta_index]
            try:
                M = -(cos(theta) / sin(theta))
                B = (p / sin(theta))
                y_0 = int(M * x_0 + B)
                y_end = int(M * x_end + B)

                cv2.line(feature_matrix, (x_0, y_0), (x_end, y_end), (255, 255, 255), 1)
                i += 1
            except ZeroDivisionError:
                y_0 = p
                y_end = p
                cv2.line(feature_matrix, (x_0, y_0), (x_end, y_end), (255, 255, 255), 1)

                i += 1

    return feature_matrix


def hough_process(feature_matrix):
    """
    Entry point into the Hough Transform method for line detection. This method essentially takes each point in the
    Hessian matrix and transforms it into a single line in the Hough space, a polar plane with axes (p, theta) instead
    of the original cartesian plane (x,y) the image as already in. This Hough space, labelled the hough_accumulator in
    the method, will increment it's values stored in it's (p, theta) coordinate for each point transformed into it's
    space.

    After each feature point in the Hessian matrix is traversed, it then obtains the highest tallied points throughout
    the Hough space and obtains it's (p, theta) points respectively. It will translate the polar coordinate back to
    the cartesian plane and draw the line detected over the feature matrix. It will also return the hough_accumulator
    in grayscale for confirmation.

    :param feature_matrix: The Hessian matrix
    :return: The detected lines drawn over the Hessian matrix as well as the Hough space accumulator
    """
    logger("3", 0, "Applying Hough Transform")

    # Indices where features exist
    row, col = np.where(feature_matrix != 0)
    
    # Create p axis
    rho_length = np.hypot(feature_matrix.shape[0], feature_matrix.shape[1])
    rho_length = int(rho_length)
    rho_linespace = np.linspace(-rho_length, rho_length, num=(2 * rho_length))

    # Create theta axis
    theta_linespace = np.linspace(0, 180, num=rho_length)
    theta_linespace = np.deg2rad(theta_linespace)
    theta_length = np.arange(len(theta_linespace))

    cos_theta = np.cos(theta_linespace)
    sin_theta = np.sin(theta_linespace)

    # Create Hough space
    hough_accumulator = np.zeros(shape=(len(rho_linespace), len(theta_linespace)))

    for y, x in zip(row, col):
        rho = (x * cos_theta) + (y * sin_theta)

        # Add rho_length to bring p to values indexable in hough space
        p = rho.astype(np.int64) + rho_length
        hough_accumulator[p, theta_length] += 1

    feature_matrix = draw_lines(hough_accumulator, feature_matrix, rho_linespace, theta_linespace)

    logger("3", 1, "Applying Hough Transform")

    return feature_matrix, hough_accumulator
