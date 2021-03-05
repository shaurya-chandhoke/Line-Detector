"""
File: line_detector.py
Author : Shaurya Chandhoke
Description: Command line script that takes an image path as input and processes the image as output
"""
import argparse
import time

import cv2
import numpy as np

from src.gaussian import gaussian_process
from src.hessian import hessian_process
from src.hough import hough_process
from src.image_output_processing import output_scaling
from src.ransac import ransac_process


def output_processing(originalImage, inputImage, gaussianImage, ransac_features, hough_features, hough_accumulator,
                      feature_matrix, timeElapsed, nosave, noshow):
    """
    The final stages of the program. This function will display the images and/or write them to files as well as
    provide an execution time.

    :param originalImage: The original image
    :param inputImage: The original image grayscaled
    :param gaussianImage: The inputImage blurred
    :param ransac_features: The inputImage with lines detected from the Ransac detector
    :param hough_features: The inputImage with lines detected form the Hough detector
    :param hough_accumulator: The hough accumulator matrix used for detecting lines from the Hough detector
    :param feature_matrix: The detected features via the Hessian feature detector
    :param timeElapsed: The total runtime time for the program
    :param nosave: Flag that determines whether to save the images to the ./out/ directory
    :param noshow: Flag that determines whether to show the images as output
    """

    if (noshow is True) and (nosave is True):
        print("(BOTH FLAGS ON) Recommend disabling either --nosave or --quiet to capture processed images")
        return 0

    print("=" * 40)
    print("Rendering Images...")

    warning = '''
    A potential divide by 0 issue was noticed while rescaling the images. 
    This may be due to a high sigma value and you may get an entirely black image. 
    It's suggested you choose a lower sigma value and try again.
    '''

    ransac_image = np.where(ransac_features == 0, inputImage, 255)
    hough_image = np.where(hough_features == 0, inputImage, 255)

    # Scaling remaining images to unsigned 8 bit integers to allow displaying and writing in grayscale
    gaussianImag, flag1 = output_scaling(gaussianImage, True)

    # Print warning message in case divide by 0 is detected
    if flag1:
        print(warning)

    if noshow is False:
        print("(DISPLAY ON) The ESC key will close all pop ups")
        cv2.imshow("Original Image", originalImage)
        cv2.imshow("Grayscale", inputImage)
        cv2.imshow("Features Detected", feature_matrix)
        cv2.imshow("Ransac Detected", ransac_image)
        cv2.imshow("Hough Detected", hough_image)
        cv2.imshow("Hough Accumulator", output_scaling(hough_accumulator))
        cv2.waitKey(0)

    if nosave is False:
        print("(IMAGE SAVE ON) Images are being written to the ./out/ folder")
        cv2.imwrite("./out/step_0_grayscale_result.jpg", inputImage)
        cv2.imwrite("./out/step_1_featuresdetected_result.jpg", feature_matrix)
        cv2.imwrite("./out/step_2_ransacdetector_result.jpg", ransac_image)
        cv2.imwrite("./out/step_3_houghdetector_result.jpg", hough_image)
        cv2.imwrite("./out/step_5_houghaccumulator_result.jpg", output_scaling(hough_accumulator))

    print("(DONE): You may want to rerun the program with the --help flag for more options to fine tune the program")
    print("=" * 40)
    print("Time to Process Image: {} seconds.".format(timeElapsed))


def start(image, sigma, hessian_threshold, ransac_threshold, ransac_points, ransac_iterations):
    """
    Starter function responsible for beginning the process for obtaining edges

    :param image: The input image as a 2D numpy array
    :param sigma: The sigma value used for generating the Gaussina filter
    :param hessian_threshold: The upper bound threshold percentage for the hessian determinant.
    :param ransac_threshold: The distance in pixels representing the threshold for the ransac line detector.
    :param ransac_points: The number of inlier points required to accept the fitted ransac line.
    :param ransac_iterations: The number of iterations to be done when looking for a line in the ransac line detector.
    :return: A series of copies of the original image passed through the detectors and filters.
    """

    print("Please wait, processing image and returning output...\n")

    gaussianImage = gaussian_process(image, sigma)
    feature_matrix = hessian_process(gaussianImage, hessian_threshold)

    # Copying the Hessian matrix so both detectors can operate independently
    feature_matrix1 = np.copy(feature_matrix)
    feature_matrix2 = np.copy(feature_matrix)

    print()

    ransac_image = ransac_process(feature_matrix1, ransac_threshold, ransac_points, ransac_iterations)

    print()

    hough_image, hough_accumulator = hough_process(feature_matrix2)

    return gaussianImage, ransac_image, hough_image, hough_accumulator, feature_matrix


def main():
    """
    Beginning entry point into the edge detection program.
    It will first perform prerequisite steps prior to starting the intended program.
    Upon parsing the command line arguments, it will trigger the start function
    """

    # Reusable message variables
    ADVICE = "rerun with the (-h, --help) for more information."

    # Start cli argparser
    temp_msg = "Given the path to an image, this program will process the image with lines detected."
    parser = argparse.ArgumentParser(prog="line_detector.py", description=temp_msg, usage="%(prog)s [imgpath] [flags]")

    temp_msg = "The file path of the image."
    parser.add_argument("imgpath", help=temp_msg, type=str)

    temp_msg = "The sigma value for Gaussian Blurring. Default value is 1."
    parser.add_argument("-gb", "--gaussian-blur", help=temp_msg, type=int, default=1)

    temp_msg = "The upper bound threshold percentage for the hessian determinant. " \
               "A higher percentage means a higher threshold (more pixels filtered). " \
               "Range is [0.0 - 1.0]. Default is 0.1."
    parser.add_argument("-ht", "--hessian-threshold", help=temp_msg, type=float, default=0.1)

    temp_msg = "The distance in pixels representing the threshold for the ransac line detector. " \
               "This value determines how wide the threshold box should be drawn when determining inliners. " \
               "Note that this value is the 1/2 the distance. " \
               "For example, a pixel value of 10 means the inlier box will be 20 pixels wide. " \
               "Default is 10 pixels."
    parser.add_argument("-rt", "--ransac-threshold", help=temp_msg, type=int, default=10)

    temp_msg = "The number of inlier points required to accept the fitted ransac line. " \
               "The higher the value, the more contrained the acceptance criteria for accepting the line. " \
               "Default is 10."
    parser.add_argument("-rp", "--ransac-points", help=temp_msg, type=int, default=10)

    temp_msg = "The number of iterations to be done when looking for a line in the ransac line detector. " \
               "Do keep in mind that a higher iteration value will most likely yield a better line at the cost of " \
               "the program's execution time. Default is 200"
    parser.add_argument("-ri", "--ransac-iteration", help=temp_msg, type=int, default=200)

    temp_msg = "The dimenions of bins that will be used to create the hough accumulator. " \
               "By default, bins will be calculated using image size. Params must be passed as: <width> <length> " \
               "in that order"
    parser.add_argument("-hgb", "--hough-bins", help=temp_msg, nargs=2, default=None)

    temp_msg = "If passed, the images will not be written to a file. By default, images are written."
    parser.add_argument("-n", "--nosave", help=temp_msg, action="store_true")

    temp_msg = "If passed, the images will not be displayed. By default, the images will be displayed."
    parser.add_argument("-q", "--quiet", help=temp_msg, action="store_true")

    # Obtain primary CLI arguments
    args = parser.parse_args()
    imgpath = args.imgpath
    nosave = args.nosave
    noshow = args.quiet

    # Obtain secondary CLI arguments for parameter tuning
    gaussian_blur = args.gaussian_blur
    hessian_threshold = args.hessian_threshold
    ransac_threshold = args.ransac_threshold
    ransac_points = args.ransac_points
    ransac_iterations = args.ransac_iteration

    # Begin error checking params and start pulling request image for processing
    originalImage = cv2.imread(imgpath)
    inputImage = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)

    if (inputImage is None) or (originalImage is None):
        print("Error: Cannot open image.\nPlease check if the path is written correctly and try again or " + ADVICE)
        return -1

    if gaussian_blur <= 0:
        print("Error: Sigma value cannot be less than 1.\nPlease try again or " + ADVICE)
        return -1

    if 1.0 < hessian_threshold or hessian_threshold < 0.0:
        print("Error: Threshold range is [0.0-1.0].\nPlease ensure the threshold is within the range or " + ADVICE)
        return -1

    np.seterr(all="ignore")
    START_TIME = time.time()
    try:
        gaussianImage, ransac_image, hough_image, hough_accumulator, feature_matrix = start(inputImage, gaussian_blur,
                                                                                            hessian_threshold,
                                                                                            ransac_threshold,
                                                                                            ransac_points,
                                                                                            ransac_iterations)
    except BaseException as e:
        print("Something went wrong, please consider re-tuning your parameters.")
        return -1

    ELAPSED_TIME = time.time() - START_TIME

    output_processing(originalImage, inputImage, gaussianImage, ransac_image, hough_image, hough_accumulator,
                      feature_matrix, ELAPSED_TIME, nosave, noshow)


if __name__ == "__main__":
    main()
