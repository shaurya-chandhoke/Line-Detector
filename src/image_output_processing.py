"""
File: image_output_processing.py
Author: Shaurya Chandhoke
Description: Helper file which contains functions used for image outputting, image saving, as well as logging.
"""
import numpy as np


def output_scaling(imageFrame, warn=False):
    """
    Global image scaling function that will take any image represented as a 2D numpy array and scale it to the
    openCV grayscale standard for image outputting. Grayscale images must be within the [0-255] range inclusively, so
    this function will scale it to that range and then convert the matrix into an 8 bit numpy array to avoid any
    potential bitwise overflow.

    If an entirely dark image (only 0 pixel values) is detected, and the warning flag is passed, it will return it's
    detected findings.

    :param imageFrame: The input image
    :param warn: Boolean indicating whether a warning flag is required to be returned
    :return: The scaled image and an optional detected black image warning flag.
    """
    scale_min = np.min(imageFrame)
    scale_max = np.max(imageFrame)
    scaled_image = np.uint8((imageFrame - scale_min) / (scale_max - scale_min) * 255)

    if warn is True:
        flag = scale_min == scale_max and scale_min == 0
        return scaled_image, flag
    else:
        return scaled_image


def logger(step, complete, message, substep=False, *argv):
    """
    Global logging function that prints the status of the program to stdout.

    :param step: The step of the program it is currently in
    :param complete: Whether the step has started or completed
    :param message: A more verbose message indicating what's happening
    :param substep: Whether this step is a child step of a larger part of the program
    :param argv: Any extra information about the step
    :return:
    """
    status = "Start" if complete == 0 else "Complete"
    finalMsg = "(Step {}) {}: {} ".format(step, status, message)

    if len(argv) != 0:
        finalMsg += "[ "
        for arg in argv:
            finalMsg += "{} ".format(arg)
        finalMsg += "]"

    if not substep:
        print(finalMsg)
    else:
        print("\t" + finalMsg)
