import os
from cv2 import cv2
import imutils
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from ClassifierHelpers import load_classifier, load_encoder, load_pca
from ContoursHelpers import draw_contour, get_contour_precedence, sort_contours

# Load the input image from disk, convert it to grayscale, and
# blur it to reduce noise
IMAGE_PATH = "images/EMNIST.png"
# IMAGE_PATH = "images/screenshot1.png"
image = cv2.imread(IMAGE_PATH)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform edge detection, find contours in the edge map, and
# sort the resulting contours from left-to-right
edged = cv2.Canny(blurred, 30, 150)
contours, hierarchy = cv2.findContours(
    edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
(contours, boundingBoxes) = sort_contours(contours, method="left-to-right")
# contours = sorted(contours, key=lambda x: get_contour_precedence(x, edged.shape[1]))

# Loop over the (now sorted) contours and draw them
contours_image = image.copy()
for i, c in enumerate(contours):
    draw_contour(contours_image, c, i)

# # Show the edged image
cv2.imshow("Edged", edged)
cv2.imshow("Contours", contours_image)
# cv2.waitKey(0)

TIMESTAMP = "20240518133504"
ENCODER_PATH = f"./checkpoint-csv/LabelEncoder-{TIMESTAMP}.pkl"
PCA_PATH = f"./checkpoint-csv/PCA-{TIMESTAMP}.pkl"
MODEL_PATH = f"./checkpoint-csv/HistGradientBoostingClassifier-{TIMESTAMP}.pkl"

encoder = load_encoder(ENCODER_PATH)
pca = load_pca(PCA_PATH)
model = load_classifier(MODEL_PATH)

detected_chars = []
for cnt, (x, y, w, h) in zip(contours, boundingBoxes):
    # Filter out bounding boxes, ensuring they are neither too small nor too large
    if (20 <= w <= 500) and (20 <= h <= 500):
        # NOTE: THRESH_BINARY_INV because values in CSV file are inverted
        # Extract the character and threshold it to make the character
        # appear as *white* (foreground) on a *black* background, then
        # grab the width and height of the thresholded image
        roi = gray[y : y + h, x : x + w]
        thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        (tH, tW) = thresh.shape

        # If the width is greater than the height, resize along the width dimension
        if tW > tH:
            thresh = imutils.resize(thresh, width=20)
        # Otherwise, resize along the height
        else:
            thresh = imutils.resize(thresh, height=20)

        # Re-grab the image dimensions (now that its been resized)
        # and then determine how much we need to pad the width and
        # height such that our image will be 32x32
        (tH, tW) = thresh.shape
        dX = int(max(0, 32 - tW) / 2.0)
        dY = int(max(0, 32 - tH) / 2.0)
        # Pad the image and force 32x32 dimensions
        padded = cv2.copyMakeBorder(
            thresh,
            top=dY,
            bottom=dY,
            left=dX,
            right=dX,
            borderType=cv2.BORDER_CONSTANT,
            value=0,  # Black
        )
        padded = cv2.resize(padded, (32, 32))  # Already dtype('uint8')
        # cv2.imshow("Test", padded)
        # cv2.waitKey(0)

        # Prepare the padded image for classification via our handwriting OCR model
        prepared = padded.flatten()

        # Update our list of characters that will be OCR'd
        detected_chars.append((prepared, (x, y, w, h)))

# Extract the bounding box locations and padded characters
boxes = [b for _, b in detected_chars]
chars = np.array([c for c, _ in detected_chars])

# Apply PCA reduction
chars = pca.transform(chars)

# OCR the characters using our handwriting recognition model
predictions = model.predict_proba(chars)

# Define the list of label names
DATASET_PATH = "dataset"
# LABELS = [chr for chr in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
LABELS = [chr for chr in encoder.classes_]

# Loop over the predictions and bounding box locations together
for pred, (x, y, w, h), idx in zip(predictions, boxes, range(len(boxes))):
    # Find top-3 indexes of the label with the largest corresponding
    # probability, then extract the probability and label
    ind = np.argpartition(pred, -3)[-3:]  # Get indexes of top-3 predictions
    i1, i2, i3 = ind[np.argsort(pred[ind])][::-1]  # Sort indexes of top-3 predictions
    p1, p2, p3 = f"{pred[i1]:.2%}", f"{pred[i2]:.2%}", f"{pred[i3]:.2%}"
    l1, l2, l3 = LABELS[i1], LABELS[i2], LABELS[i3]
    # Draw only top-1 prediction on the image
    print(f"[INFO] {idx+1:>3}: {l1:>3} - {p1:>6}, {l2:>3} - {p2:>6}, {l3:>3} - {p3:>6}")
    color = (36, 255, 12)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, l1, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# Show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
