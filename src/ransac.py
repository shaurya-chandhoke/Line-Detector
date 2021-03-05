"""
File: ransac.py
Author: Shaurya Chandhoke
Description: Helper file which contains the function used for line detection via the Ransac method
"""
from src.image_output_processing import logger

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ransac_error(coefficients, x_i, y_i, testVal):
    """
    Helper function that will evalutate the residuals of the function. This will be determined using the
    Total Least Squares method of calculating residuals:

    The model is converted from slope intercept form (y = mx + b) into standard form (Ax + By = d; A > 0)

    :param coefficients: The model's standard form coefficients
    :param x_i: Points in the image to test the model in the x-axis
    :param y_i: Points in the image to test the model in the y-axis
    :param testVal: The total number of true values. This is used in computing the error function.
    :return: The residual value.
    """
    A = coefficients[0]
    B = coefficients[1]
    n = len(testVal)

    # Computing error of proposed line (using total least square residual algorithm to evaluate normal residual)
    # Means
    x_bar = (A / n) * (np.sum(x_i))
    y_bar = (B / n) * (np.sum(y_i))

    # Cost in x, y components
    E_x = A * (x_i - x_bar)
    E_y = B * (y_i - y_bar)

    # Total cost is summation of components squared to remove negative sign
    E = np.sum(np.square(E_x + E_y))

    return E


def inlierCounter(feature_matrix, upperModel, lowerModel):
    """
    Helper function that will count how many inliers are within the model's threshold box.

    :param feature_matrix: The Hessian matrix
    :param upperModel: The upper bound model's coefficients
    :param lowerModel: The lower bound model's coefficients
    :return: The number of inliers as well as the coordinates of those inliers
    """
    m_upper = upperModel[0]
    b_upper = upperModel[1]

    m_lower = lowerModel[0]
    b_lower = lowerModel[1]

    inlier_threshold = b_upper - b_lower
    inlier_coordinates_x = []
    inlier_coordinates_y = []
    inlier_counter = 0
    row, col = np.where(feature_matrix != 0)

    for y, x in zip(row, col):
        model_upper = (m_upper * x) + b_upper
        model_lower = (m_lower * x) + b_lower

        if abs(y - model_upper) < inlier_threshold and abs(y - model_lower) < inlier_threshold:
            inlier_counter += 1
            inlier_coordinates_x.append(x)
            inlier_coordinates_y.append(y)

    return inlier_counter, inlier_coordinates_x, inlier_coordinates_y


def ransac_process(feature_matrix, delta_threshold, inlier_threshold, iterations):
    """
    Entry point into the Ransac detection method. This method will run a series of iteration in which it will randomly
    select two points in the image and create a model from it. It will then assess how contaminated the model is.

    When the model is generated, there is a delta_threshold assigned to it. The delta_threshold essentially generates
    two more models parallel to the original model. In other words, the delta_threshold determines the parallel model's
    y-intercept. The reason for this is to create a 'rectangular box' around the model to determine which inliers are
    found within the model:

        upper bound model->     ------------------
                                        | delta_threshold
        original model->        ==================
                                        | delta_threshold
        lower bound model->     ------------------

    After the series of iteration it will pick the four best models and draw the models over the feature matrix.
    It will also transform the image to a matplotlib scatter plot with the models drawn for a more detailed view.


    :param feature_matrix: The Hessian feature matrix
    :param delta_threshold: For each model generated, a rectangular box is draw alongside it. This represents its width.
    :param inlier_threshold: The minimum number of points required to accept the model
    :param iterations: The number of iterations to run the detector for. High number = better results = longer run time.
    :return: The feature matrix with the model lines drawn over it.
    """
    logger("2", 0, "Ransac Line Detection")

    row, col = np.where(feature_matrix != 0)
    plotFrame = pd.DataFrame({"x": col, "y": row, "val": feature_matrix[row, col]})

    trueX = plotFrame.loc[:, "x"]
    trueY = plotFrame.loc[:, "y"]
    trueVal = plotFrame.loc[:, "val"]

    # Generate scatter plot transformation from image to cartesian points
    plt.scatter(data=plotFrame, x="x", y="y", marker=".")
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.gca().invert_yaxis()

    horizontalAxis = np.arange(xlim[0], xlim[1])

    inlierTracker = []
    inlierCoords = []
    modelTracker = []
    modelErrorTracker = []

    for i in range(iterations):
        choice = plotFrame.sample(n=2)
        x1, y1 = (choice.iloc[0, 0], choice.iloc[0, 1])
        x2, y2 = (choice.iloc[1, 0], choice.iloc[1, 1])

        m = (y2 - y1) / (x2 - x1)
        b = y1 - (m * x1)

        A = (y2 - y1) / (x2 - x1)
        B = 1 / A

        standard_form_coeff = (-A, -B) if A < 0 else (A, B)
        error = ransac_error(standard_form_coeff, trueX, trueY, trueVal)
        inliers, coordsX, coordsY = inlierCounter(feature_matrix, (m, b + delta_threshold), (m, b - delta_threshold))

        inlierTracker.append(inliers)
        inlierCoords.append((coordsX, coordsY))
        modelTracker.append((m, b))
        modelErrorTracker.append(error)

    inlierTracker = np.asarray(inlierTracker)
    condition = inlierTracker > inlier_threshold

    inlierTracker = np.extract(condition, inlierTracker)

    sortedCounts = np.sort(inlierTracker)
    sortedCounts = np.unique(np.sort(sortedCounts))
    sortedCounts = sortedCounts[::-1]

    i = 0

    while i != 4 and i < len(sortedCounts):
        highestCount = sortedCounts[i]
        indices = np.where(inlierTracker == highestCount)

        for index in np.nditer(indices):
            if i != 4 and i < len(sortedCounts):
                xc = inlierCoords[index][0]
                yc = inlierCoords[index][1]
                model_m = modelTracker[index][0]
                model_b = modelTracker[index][1]

                plt.plot(horizontalAxis, (model_m * horizontalAxis) + model_b, linestyle="-", c="g")
                plt.scatter(xc, yc, marker="D", c="r", s=10)

                cv2.line(feature_matrix, (xc[0], yc[0]), (xc[-1], yc[-1]), (255, 255, 255), 1)

                i += 1

    plt.title("Top 4 Strongest Lines After {} Iterations".format(iterations))
    plt.savefig("./out/step_2_ransacdetector_result_scatterplot.png")

    logger("2", 1, "Ransac Line Detection. Scatter plot has been saved to ./out/step2_scatterplot.png")

    return feature_matrix
