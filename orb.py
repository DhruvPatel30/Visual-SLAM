import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys

image = cv2.imread("img_2.jpg")
dimension = (480,640)

if image is None:
    print("NO IMAGE !!!")
    sys.exit()

image = cv2.resize(image, dimension, interpolation = cv2.INTER_AREA)

orb = cv2.ORB_create()      # Initiate ORB detector

keypoints = orb.detect(image,None)   # find the keypoints with ORB

keypoints, descriptor = orb.compute(image, keypoints)  # compute the descriptors with ORB

image2 = cv2.drawKeypoints(image, keypoints, None, color=(0,255,0), flags=0)

plt.imshow(image2)
plt.show()