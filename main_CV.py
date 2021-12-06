import cv2
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from numpy.core.numeric import cross


def get_orb_features(left_image_gray, right_image_gray, left_image, right_image):
    # initiate orb detector
    orb = cv2.ORB_create()

    # Detect keytpoints and descriptors
    kp_l, des_l = orb.detectAndCompute(left_image_gray,None)
    kp_r, des_r = orb.detectAndCompute(right_image_gray,None)

    # initiate the matcher
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    # Match descriptors.
    matches = bf_matcher.knnMatch(des_l,des_r, k=2)

    good = []
    pts1 = []
    pts2 = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            good.append(m)
            pts2.append(kp_r[m.trainIdx].pt)
            pts1.append(kp_l[m.queryIdx].pt)

    pts2 = np.float32(pts2)
    pts1 = np.float32(pts1)
            
        # x = np.array(np.float32(x))
        # x_dash = np.array(np.float32(x_dash))

        # img3 = cv2.drawMatchesKnn(left_image,kp_l,right_image,kp_r,refined_match,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # plt.imshow(img3)
        # plt.show()
    return pts1, pts2, good

def cal_epipolar_constraint_equation( temp_x, temp_x_dash, F):
    temp_x_dash_transpose = np.transpose(temp_x_dash)
    # print(temp_x_dash_transpose)
    temp1 = np.dot(F, temp_x)
    final = np.dot(temp_x_dash_transpose, temp1)
    # print(final)
    return final


def main():
    # Reading images
    left_image = cv2.imread("AprilCalib_1.png")
    right_image = cv2.imread("AprilCalib_2.png")

    # Checking if the image is properly read   
    if left_image is None:
        print("LEFT IMAGE NOT DETECTED!!!")
        sys.exit()    
    if right_image is None:
        print("RIGHT IMAGE NOT DETECTED!!!")
        sys.exit()

    # resizing the image
    # dimension = (960,1000)
    # left_image  = cv2.resize(left_image , dimension, interpolation = cv2.INTER_AREA)
    # right_image  = cv2.resize(right_image , dimension, interpolation = cv2.INTER_AREA)

    # converting the images to gray scale
    left_image_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_image_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    x, x_dash, refined_match = get_orb_features(left_image_gray, right_image_gray, left_image, right_image)
    print("x", len(x))
    # print("x_dash", x_dash)
    # print(refined_match)

    F, mask = cv2.findFundamentalMat(x, x_dash, cv2.FM_RANSAC, 0.0001, 0.99)
    print(F)

    # We select only inlier points
    x = x[mask.ravel() == 1]
    x_dash = x_dash[mask.ravel() == 1]
    print("x", len(x))

    for i in range(len(x)- 1):
        image = cv2.circle(left_image, (x[i][0],x[i][1]), radius=2, color=(0, 0, 255), thickness=2)
        imager = cv2.circle(right_image, (x_dash[i][0],x_dash[i][1]), radius=2, color=(0, 0, 255), thickness=2)
    cv2.imshow('feature_matched',image)
    cv2.imshow('matched',imager)
    cv2.waitKey(0)
    


if __name__ == "__main__":
    main()