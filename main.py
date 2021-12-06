import cv2
import numpy as np
import math
import sys
import random
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


class calculate_F_matrix():
    def __init__(self):

        # reading the images.
        self.left_image = cv2.imread("random_l.png")
        self.right_image = cv2.imread("random_r.png")

        # print(self.left_image.shape)
        # print(self.right_image.shape)

        # checking if the image has been read properly 
        if self.left_image is None:
            print("LEFT IMAGE NOT DETECTED!!!")
            sys.exit()
        
        if self.right_image is None:
            print("RIGHT IMAGE NOT DETECTED!!!")
            sys.exit()

        # resizing the image
        # self.dimension = (960,1000)
        # self.left_image  = cv2.resize(self.left_image , self.dimension, interpolation = cv2.INTER_AREA)
        # self.right_image  = cv2.resize(self.right_image , self.dimension, interpolation = cv2.INTER_AREA)

        # converting the image to grayscale
        self.left_image_gray = cv2.cvtColor(self.left_image, cv2.COLOR_BGR2GRAY)
        self.right_image_gray = cv2.cvtColor(self.right_image, cv2.COLOR_BGR2GRAY)

    def match_features(self):

        #initialize the ORB detector
        orb = cv2.ORB_create(nfeatures = 1000)
        
        #detect key points and compute descriptors
        kp_l, des_l = orb.detectAndCompute(self.left_image_gray, None)
        kp_r, des_r = orb.detectAndCompute(self.right_image_gray, None)
        
        #image with key points detected
        kp_img_l = cv2.drawKeypoints(self.left_image, kp_l, None, color=(0, 255, 0), flags=0)
        kp_img_r = cv2.drawKeypoints(self.right_image, kp_r, None, color=(0, 255, 0), flags=0)
        
        #matching features
        brute_force = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = brute_force.knnMatch(des_l,des_r,k=2)
        # matches = brute_force.match(des_l, des_r)
        # matches=sorted(matches, key= lambda x:x.distance)

        # print(len(kp_l))
        print("matches", len(matches))
        # print(len(matches))

        #lowe's ratio method
        refined_match = []
        x = []
        x_dash = []
        for m,n in matches:
            if m.distance < 0.75 * n.distance:
                refined_match.append([m])
                # print(m.trainIdx)
                x.append(kp_l[m.queryIdx].pt)
                x_dash.append(kp_r[m.trainIdx].pt)
                
        x = np.array(np.float32(x))
        x_dash = np.array(np.float32(x_dash))

        # print("F matrix from CV")
        # F, mask = cv2.findFundamentalMat(x,x_dash,cv2.FM_LMEDS)
        # print(F)
        # print("x", len(x))
        # print("x_dash", len(x_dash))
        print("refined_match", len(refined_match))
        matched_img = cv2.drawMatchesKnn(self.left_image,kp_l,self.right_image,kp_r,refined_match,None,flags=2)
        
        # cv2.imshow('ORB_L', kp_img_l)
        # cv2.imshow('ORB_R', kp_img_r)
        
        #image = cv2.circle(self.img_left, (x[0][0],x[0][1]), radius=5, color=(0, 0, 255), thickness=2)
        #imager = cv2.circle(self.img_right, (x_dash[0][0],x_dash[0][1]), radius=5, color=(0, 0, 255), thickness=2)
        #cv2.imshow('feature_matched',image)
        #cv2.imshow('matched',imager)
        # cv2.imshow('feature_matched',matched_img)
        # cv2.waitKey(0)
        return x, x_dash
        
    def F_matrix(self, x, x_dash):         # USING NORMALIZED 8 - POINT ALGORITHM TO COMPUTE F MATRIX 

        # The key to success with the 8-point algorithm is proper careful normalization of the input data before constructing the equations to solve.
        # REFERENCE - https://www.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj3/html/sdai30/index.html

        # -----------------------------------------------------------------------------------------------------------
        # STEP 1 - Computing the centroid of the feature points 
        # computing the centroid of the feature points in left image (self.x)
        u_total = 0
        v_total = 0
        for u,v in x:
            # print(u,v)
            u_total += u
            v_total += v
        u_bar = u_total / len(x)
        v_bar = v_total / len(x)
        
        # computing the centroid of the feature points in right image (self.x_dash)
        u_dash_total = 0
        v_dash_total = 0
        for u,v in x_dash:
            u_dash_total += u
            v_dash_total += v
        u_dash_bar = u_dash_total / len(x_dash)
        v_dash_bar = v_dash_total / len(x_dash)
        # --------------------------------------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------------------------------------
        # STEP 2 - Recentering by substracting the mean u_bar and v_bar from the original coordinates (u,v) in self.x
        u_recenter = []
        v_recenter = []
        for u, v in x:
            u_recenter.append(u-u_bar)
            v_recenter.append(v-v_bar)

        # recentering by substracting the mean u_dash_bar and v_dash_bar from the original coordinates (u,v) in self.x_dash
        u_dash_recenter = []
        v_dash_recenter = []
        for u, v in x_dash:
            u_dash_recenter.append(u-u_dash_bar)
            v_dash_recenter.append(v-v_dash_bar)
        # ----------------------------------------------------------------------------------------------------------------

        # ----------------------------------------------------------------------------------------------------------------
        # STEP 3 - Defining scale term s and s_dash
        # Defining s
        denominator_s = 0
        for i in range(len(u_recenter)):            # taking values from u_receneter and v_recenter in one for loop
            denominator_s = denominator_s + (u_recenter[i]**2 + v_recenter[i]**2)
        denominator_s = denominator_s / len(u_recenter)
        s = math.sqrt(2) / math.sqrt(denominator_s)

        # Defining s_dash
        denominator_s_dash = 0
        for i in range(len(u_dash_recenter)):            # taking values from u_receneter and v_recenter in one for loop
            denominator_s_dash = denominator_s_dash + (u_dash_recenter[i]**2 + v_dash_recenter[i]**2)
        denominator_s_dash = denominator_s_dash / len(u_dash_recenter)
        s_dash = math.sqrt(2) / math.sqrt(denominator_s_dash)

        # ----------------------------------------------------------------------------------------------------------------

        # ----------------------------------------------------------------------------------------------------------------
        #  STEP 4 - Constructing the Transformation Matrix
        # T
        T1 = np.array([[s,0,0],[0,s,0],[0,0,1]])
        T2 = np.array([[1,0,-u_bar],[0,1,-v_bar],[0,0,1]])
        T = np.dot(T1,T2)

        # T_dash
        T1_dash = np.array([[s_dash,0,0],[0,s_dash,0],[0,0,1]])
        T2_dash = np.array([[1,0,-u_dash_bar],[0,1,-v_dash_bar],[0,0,1]])
        T_dash = np.dot(T1_dash,T2_dash)

        # ----------------------------------------------------------------------------------------------------------------

        # ----------------------------------------------------------------------------------------------------------------
        # STEP 5 - Computation to obtain normalized points.
        x_normalized = []
        x_dash_normalized = []
        for i in range(len(x)):                    # taking values from self.x and self.x_dash to covert into normalized points, in one for loop
            temp = np.array([[x[i][0]],[x[i][1]], [1]])
            x1 = np.dot(T, temp)
            x_normalized.append(x1)

            temp_ = np.array([[x_dash[i][0]],[x_dash[i][1]], [1]])
            x1_ = np.dot(T_dash, temp_)
            x_dash_normalized.append(x1_)

        # ----------------------------------------------------------------------------------------------------------------

        # ----------------------------------------------------------------------------------------------------------------
        # #compute A matrix to solve Af=0 equation and perform SVD on A 
        A = np.zeros((len(x),9))
        for i in range(len(x)):
            A[i] = [x_dash_normalized[i][0]*x_normalized[i][0], x_dash_normalized[i][0]*x_normalized[i][1], x_dash_normalized[i][0], \
                    x_dash_normalized[i][1]*x_normalized[i][0], x_dash_normalized[i][1]*x_normalized[i][1], x_dash_normalized[i][1], \
                    x_normalized[i][0], x_normalized[i][1], 1 ]

        #solve SVD on A
        U,S,VT=np.linalg.svd(A,full_matrices=True)
        # print(VT, "vvv", VT.shape)
        #F_norm matrix from the column of V corresponding to the least singular value.
        #rearrange the 9 entries of f to create the 3x3 fundamental matrix F_norm
        F_norm = VT[8,:].reshape(3,3)
        # F_norm = VT[-1]
        # F_norm = np.reshape(F_norm, (3,3)) 
        #SVD on F_norm
        Uf,Sf,VTf = np.linalg.svd(F_norm, full_matrices=True)
        # print("sss", Sf)
        #smallest singular value in Sf is changed to 0
        Sf[2] = 0
        S_matrix = np.diag(Sf)
        #(constraint enforcement)F matrix after making smallest singular value 0 to recompute the rank from 3 to 2.
        F_norm = np.dot(Uf, np.dot( S_matrix, VTf))
        det = np.linalg.det(F_norm) #will be equal to 0
        #computing the original F matrix
        F_orig = np.dot(np.transpose(T_dash), np.dot(F_norm,T))
        # F_orig = np.around(F_orig, decimals=4)
        # print("----- F Matrix -----")
        # print(F_orig)
        return F_orig
        # ----------------------------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------------------------
    # calculate E matrix
    def calculate_E_matrix(self,F, K):
        temp_K_transpose = np.transpose(K)
        temp1 = np.dot(F, K)
        E = np.dot(temp_K_transpose, temp1)
        return E
    # ----------------------------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------------------------
    def decompose_E_matrix(self, E):
        # Decompose Essential Matrix
        U, S, VT = np.linalg.svd(E,full_matrices=True) 
        S = np.array([[1,0,0],[0,1,0],[0,0,0]])
        W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)

        #translation vector
        # t1 = U[:,2]
        # t2 = -U[:,2]
        t1 = np.array([[U[0][2]], [U[1][2]], [U[2][2]]])
        t2 = -t1
        print("t1", t1)
        #rotation vector
        R1 = np.dot(np.dot(U,W),VT)
        R2 = np.dot(np.dot(U,np.transpose(W)),VT)
        #print(R1,R2)
        return t1,t2,R1,R2
    # ----------------------------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------------------------
    # checking cheriality condition to deal with the ambiguity, and to find correct unique camera pose
    def cheriality(self,t, C, R, x_inlier, x_dash_inlier, K):
        # t = np.transpose(t)
        extrinsics = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])      # K[R|t]. K is intrinsics and [R|t] is extrinsics. For the first frame considering R=I and t=0
        extrinsics_dash = np.array([[R[0][0], R[0][1],R[0][2],t[0]], [R[1][0],R[1][1],R[1][2],t[1]], [R[2][0],R[2][1],R[2][2],t[2]]])

        P = np.dot(K,extrinsics)
        P_dash = np.dot(K,extrinsics_dash)

        X_3d = cv2.triangulatePoints(projMatr1= np.float32(P), projMatr2=np.float32(P_dash) ,projPoints1=np.transpose(x_inlier), projPoints2=np.transpose(x_dash_inlier))
        X_3d = X_3d/X_3d[3]                     # Need to understand this line
        X_3d = np.delete(X_3d, 3, 0)            # Need to understand this line
        X_3d = np.transpose(X_3d)
        # print(len(X_3d))
        X_3d_front = []   
        count = 0
        # print("C",C)
        # print("X_3d", X_3d[0])

        for point_3d in X_3d:
            
            X_minus_C = [point_3d[0]- C[0], point_3d[1]- C[1], point_3d[2]- C[2]]       # column matrix
            # X_minus_C_ = np.array([[X_minus_C[0]], [X_minus_C[1]], [X_minus_C[2]]])
            r2 = R[2, :]
            # r2 = np.array(r2)
            # print(X_minus_C.shape)
            # print(r2.shape)
            # print(X_minus_C)
            # print(r2)
            depth_Z = np.dot(r2, X_minus_C)
            if depth_Z > 0:
                X_3d_front.append(point_3d)
                count += 1
        return count, X_3d_front

    # ----------------------------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------------------------
    # calculating R and t.
    def estimate_pose(self,t1,t2,R1,R2,x_inlier,x_dash_inlier, K):

        # four possible configuration of t and R
        t_1 = t1
        t_2 = t2
        t_3 = t1
        t_4 = t2

        R_1 = R1
        R_2 = R1
        R_3 = R2
        R_4 = R2 

        if np.linalg.det(R_1) < 0:
            R_1 = -R_1
            t_1 = -t_1
        if np.linalg.det(R_2) < 0:
            R_2 = -R_2
            t_2 = -t_2
        if np.linalg.det(R_3) < 0:
            R_3 = -R_3
            t_3 = -t_3
        if np.linalg.det(R_4) < 0:
            R_4 = -R_4
            t_4 = -t_4

        # t = -RC
        C_1 = -np.dot(np.transpose(R_1), t_1)
        C_2 = -np.dot(np.transpose(R_2), t_2)
        C_3 = -np.dot(np.transpose(R_3), t_3)
        C_4 = -np.dot(np.transpose(R_4), t_4)
        
        count1, X_3d_front1 = self.cheriality(t_1,C_1, R_1, x_inlier, x_dash_inlier, K)
        count2, X_3d_front2 = self.cheriality(t_2, C_2, R_2, x_inlier, x_dash_inlier, K)
        count3, X_3d_front3 = self.cheriality(t_3, C_3, R_3, x_inlier, x_dash_inlier, K)
        count4, X_3d_front4 = self.cheriality(t_4, C_4, R_4, x_inlier, x_dash_inlier, K)

        # print("count1", count1, len(X_3d_front1))
        # print("count2", count2, len(X_3d_front2))
        # print("count3", count3, len(X_3d_front3))
        # print("count4", count4, len(X_3d_front4))

        final_count = max(count1, count2, count3, count4)
        if final_count == count1:
            return X_3d_front1, R_1, t_1, C_1
        elif final_count == count2:
            return X_3d_front2, R_2, t_2, C_2
        elif final_count == count3:
            return X_3d_front3, R_3, t_3, C_3
        elif final_count == count4:
            return X_3d_front4, R_4, t_4, C_4
        
    # ----------------------------------------------------------------------------------------------------------------
        
        
    # ----------------------------------------------------------------------------------------------------------------
    # visualize 3d points
    def visualize_3d_points(self, X_3d, t_final, C_final):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # print(len(X_3d), "lll")
        # print("C", C_final)
        print(X_3d[0])
        for point in X_3d:
            x = point[0]
            y = point[1]
            z = point[2]
            ax.scatter(x, y, z, c='r', marker='o')
        
        ax.scatter(C_final[0], C_final[1], C_final[2], c='g', marker='o')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()
        # print(X_3d)

    # ----------------------------------------------------------------------------------------------------------------
    # Function to calulate epipolar eqaution (xdash' . F. x) to get the error value, which is compared with the threshold to calculate inliers
    def cal_epipolar_constraint_equation(self, temp_x, temp_x_dash, F):
        temp_x_dash_transpose = np.transpose(temp_x_dash)
        # print(temp_x_dash_transpose)
        temp1 = np.dot(F, temp_x)
        final = np.dot(temp_x_dash_transpose, temp1)
        # print(final)
        return final
    # ----------------------------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------------------------
    # Function to calculate the inliers.
    def calculate_inliers(self, F, x ,x_dash):
        inliers = 0
        # x_inliers = []
        # x_dash_inliers = []
        for i in range(len(x)):
            temp_x = np.array(([x[i][0]], [x[i][1]], [1]))                          # 3*1 array
            temp_x_dash = np.array(([x_dash[i][0]], [x_dash[i][1]], [1]))           # 3*1 array
            threshold = 0.001
            if abs(self.cal_epipolar_constraint_equation(temp_x, temp_x_dash, F)) < threshold:        # Need to confirm if abs need to be added or not
                # x_inliers.append([x[i][0], x[i][1]])
                # x_dash_inliers.append([x_dash[i][0], x_dash[i][1]])
                inliers += 1
        return inliers
    # ----------------------------------------------------------------------------------------------------------------


    def ransac(self):           # to robustly estimate the F matrix. Will compute inliers and outliers
        # ----------------------------------------------------------------------------------------------------------------
        x, x_dash = self.match_features()
        # print("x",x)
        # print("x_dash",x_dash)
        N = 1000                 # no. of iterations of ransac algorithm
        max_inliers = 0
        # x_inliers_final = []
        # x_dash_inliers_final = []
        high_score_F =np.zeros((3,3))
        for i in range(N):
            x_random = []
            x_dash_random = []
            for j in range(8):              # select 8 pairs of correspondance and compute F matrix
                random_int = random.randint(0, len(x)-1)
                x_random.append(x[random_int])
                x_dash_random.append(x_dash[random_int])
            x_random = np.array(x_random)
            x_dash_random = np.array(x_dash_random)
            # print(x_random)
            # print(x_dash_random)

            F  = self.F_matrix(x_random, x_dash_random)
            total_inliers = self.calculate_inliers(F, x, x_dash)
            if total_inliers > max_inliers:
                max_inliers = total_inliers
                high_score_F = F
                # x_inliers_final = x_inliers
                # x_dash_inliers_final = x_dash_inliers

        print("max_inliers",max_inliers)
        print("high_score_F")
        print(high_score_F)
        # ----------------------------------------------------------------------------------------------------------------

        # ----------------------------------------------------------------------------------------------------------------
        # Taking high_score_F as the final F matrix and finding the possible inliers
        x_inliers = []
        x_dash_inliers = []
        for i in range(len(x)):
            temp_x = np.array(([x[i][0]], [x[i][1]], [1]))                          # 3*1 array
            temp_x_dash = np.array(([x_dash[i][0]], [x_dash[i][1]], [1]))           # 3*1 array
            threshold = 0.001
            if abs(self.cal_epipolar_constraint_equation(temp_x, temp_x_dash, high_score_F)) < threshold:    # Need to confirm if abs need to be added or not
                x_inliers.append([x[i][0], x[i][1]])
                x_dash_inliers.append([x_dash[i][0], x_dash[i][1]])

        x_inliers = np.array(x_inliers)
        x_dash_inliers = np.array(x_dash_inliers)

        print(len(x_inliers), "xxxxxxx")
        # print(x_dash_inliers)
        # print("X_inliers",x_inliers)
        # print("x_dash_inliers",x_dash_inliers)

        for i in range(len(x_inliers)- 1):
            image = cv2.circle(self.left_image, (x_inliers[i][0],x_inliers[i][1]), radius=1, color=(0, 0, 255), thickness=2)
            imager = cv2.circle(self.right_image, (x_dash_inliers[i][0],x_dash_inliers[i][1]), radius=1, color=(0, 0, 255), thickness=2)
        cv2.imshow('feature_matched',image)
        cv2.imshow('matched',imager)
        cv2.waitKey(0)
        # ----------------------------------------------------------------------------------------------------------------

        # ----------------------------------------------------------------------------------------------------------------
        # calculate Essential Matrix. E =(Kl)'. F. Kr
        # K = np.array(([1520.4 , 0, 302.32], [0 , 1522.9, 246.87], [0 , 0, 1] ))   # temple
        K = np.array(([535.4 , 0, 320.1], [0 , 539.2, 247.6], [0 , 0, 1] )) # table chair
        E = self.calculate_E_matrix(F,K)
        print("Essential Matrix")
        print(E)
        # ----------------------------------------------------------------------------------------------------------------

        # ----------------------------------------------------------------------------------------------------------------
        t1, t2, R1, R2 = self.decompose_E_matrix(E)

        X_3d, R_final, t_final, C_final = self.estimate_pose(t1,t2,R1,R2,x_inliers,x_dash_inliers, K)

        self.visualize_3d_points(X_3d, t_final, C_final)

        # ----------------------------------------------------------------------------------------------------------------


def main():
    
    F_matrix = calculate_F_matrix()
    # x , x_dash = F_matrix.match_features()
    # F_matrix.F_matrix(x, x_dash)
    F_matrix.ransac()

if __name__ == "__main__":
    main()