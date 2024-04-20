from cv2 import cv2
import numpy as np


def sort_contours(cnts, method="left-to-right"):
    # Initialize the reverse flag and sort index
    reverse = False
    i = 0

    # Handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # Handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # Construct the list of bounding boxes and sort them from top to bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(
        *sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse)
    )

    # Return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def draw_contour(image, c, i):
    color = (255, 236, 161)

    # Compute the center of the contour area and draw a circle representing the center
    # https://stackoverflow.com/questions/62392240/opencv-cv2-moments-returns-all-moments-to-zero
    # M = cv2.moments(c)
    # cX = int(M["m10"] / M["m00"])
    # cY = int(M["m01"] / M["m00"])
    x, y, w, h = cv2.boundingRect(c)
    image = cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    # Draw the countour number on the image
    # cv2.putText(
    #     image,
    #     f"{i + 1}",
    #     (x - 5, y - 5),
    #     cv2.FONT_HERSHEY_COMPLEX,
    #     0.5,  # Font size
    #     color,
    #     1,  # Thickness
    # )

    # Return the image with the contour number drawn on it
    return image


def get_contour_precedence(contour, cols):
    tolerance_factor = 30
    x, y, w, h = cv2.boundingRect(contour)
    return ((y // tolerance_factor) * tolerance_factor) * cols + x
