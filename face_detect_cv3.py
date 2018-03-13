#from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import cv2
import sys
import argparse
import imutils
import numpy as np
import time


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--test_images", required=True, help="path to images directory")
args = vars(ap.parse_args())


start = time.time()
# Get user supplied values
#imagePath = sys.argv[1]
cascPath = "facial_recognition_model.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)


# loop over the image paths
imagePaths = list(paths.list_images(args["test_images"]))
count = 0

for imagePath in imagePaths:
	# load the image and resize it to (1) reduce detection time
	# and (2) improve detection accuracy
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=min(400, image.shape[1]))
	#orig = image.copy()
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Detect faces in the image
	faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
            #flags = cv2.CASCADE_SCALE_IMAGE
            )
	print("Found {0} faces!".format(len(faces)))
	if len(faces) > 0:
            count = count + len(faces)
	#in this code i excluded the overlapping
	# Draw a rectangle around the faces
#	for (x, y, w, h) in faces:
#            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
#            cv2.imshow("Faces found", image)
#            cv2.waitKey(0)

print("Found", count, "objects!")
end = time.time()
print(end - start)
