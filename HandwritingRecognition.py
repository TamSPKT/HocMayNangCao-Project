import os
from cv2 import cv2
from keras import saving
import imutils
import numpy as np

from ContoursHelpers import draw_contour, get_contour_precedence, sort_contours

# Load the input image from disk, convert it to grayscale, and
# blur it to reduce noise
IMAGE_PATH = "images/screenshot1.png"
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

detected_chars = []
for cnt, (x, y, w, h) in zip(contours, boundingBoxes):
    # Filter out bounding boxes, ensuring they are neither too small nor too large
    if (20 <= w <= 500) and (20 <= h <= 500):
        # Extract the character and threshold it to make the character
        # appear as *black* (foreground) on a *white* background, then
        # grab the width and height of the thresholded image
        roi = gray[y : y + h, x : x + w]
        thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        (tH, tW) = thresh.shape

        # If the width is greater than the height, resize along the width dimension
        if tW > tH:
            thresh = imutils.resize(thresh, width=96)
        # Otherwise, resize along the height
        else:
            thresh = imutils.resize(thresh, height=96)

        # Re-grab the image dimensions (now that its been resized)
        # and then determine how much we need to pad the width and
        # height such that our image will be 128x128
        (tH, tW) = thresh.shape
        dX = int(max(0, 128 - tW) / 2.0)
        dY = int(max(0, 128 - tH) / 2.0)
        # Pad the image and force 128x128 dimensions
        padded = cv2.copyMakeBorder(
            thresh,
            top=dY,
            bottom=dY,
            left=dX,
            right=dX,
            borderType=cv2.BORDER_CONSTANT,
            value=255,  # White
        )
        padded = cv2.resize(padded, (128, 128))
        # cv2.imshow("Test", padded)
        # cv2.waitKey(0)

        # Prepare the padded image for classification via our handwriting OCR model
        prepared = padded.astype("float32")
        prepared = np.expand_dims(prepared, axis=-1)

        # Update our list of characters that will be OCR'd
        detected_chars.append((prepared, (x, y, w, h)))

# Extract the bounding box locations and padded characters
boxes = [b for _, b in detected_chars]
chars = np.array([c for c, _ in detected_chars], dtype="float32")

# TODO: Nested model causes loading failed.
model = saving.load_model("checkpoint\\model-EfficientNet-20240419122050.keras")
# model.summary()  # type: ignore

# OCR the characters using our handwriting recognition model
predictions = model.predict(chars, batch_size=len(chars))  # type: ignore

# Define the list of label names
DATASET_PATH = "dataset"
LABELS = [
    filename
    for filename in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, filename))
]

# Loop over the predictions and bounding box locations together
for pred, (x, y, w, h), idx in zip(predictions, boxes, range(len(boxes))):
    # Find top-3 indexes of the label with the largest corresponding
    # probability, then extract the probability and label
    ind = np.argpartition(pred, -3)[-3:]  # Get indexes of top-3 predictions
    i1, i2, i3 = ind[np.argsort(pred[ind])][::-1]  # Sort indexes of top-3 predictions
    p1, p2, p3 = pred[i1], pred[i2], pred[i3]
    l1, l2, l3 = LABELS[i1], LABELS[i2], LABELS[i3]
    # Draw only top-1 prediction on the image
    print(f"[INFO] {idx+1}: {l1} - {p1:.2%}, {l2} - {p2:.2%}, {l3} - {p3:.2%}")
    color = (36, 255, 12)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, l1, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# Show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)