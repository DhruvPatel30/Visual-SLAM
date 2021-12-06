import numpy as np
import cv2
import matplotlib.pyplot as plt

image_1 = cv2.imread("img_1.jpg")
image_2 = cv2.imread("img_2.jpg")

orb = cv2.ORB_create()

keypoints_1, descriptors_1 = orb.detectAndCompute(image_1, None)
keypoints_2, descriptors_2 = orb.detectAndCompute(image_2, None)


# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
bf = cv2.BFMatcher()

# matches = bf.match(descriptors_1, descriptors_2)
matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)

# matches = sorted(matches, key= lambda x:x.distance)
# print(matches)
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# image_3 = cv2.drawMatches(image_1,keypoints_1,image_2,keypoints_2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
image_3 = cv2.drawMatchesKnn(image_1,keypoints_1,image_2,keypoints_2,good,None,flags=2)

plt.imshow(image_3)
plt.show()