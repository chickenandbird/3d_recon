# perform complete 3D reconstruction from 11 images
import bundle_adjustment
import feature_extraction
import feature_matching
import initial_recon
import pnp_recon
import numpy as np
import cv2 
def normalize_points(points):
    homo_points = np.hstack((points, np.ones((points.shape[0], 1))))
    mean = np.mean(homo_points[:, :-1], axis=0)
    scale = np.sqrt(2) / np.std(homo_points[:, :-1], axis=0)
    T = np.diag([scale[0], scale[1], 1])
    T[:-1, -1] = -mean * scale
    norm_points = np.dot(T, homo_points.T).T
    return norm_points, T

def DLT(src_norm, dst_norm,T_src,T_dst):
    A = []
    for i in range(src_norm.shape[0]):
        x, y, _ = src_norm[i]
        xp, yp, _ = dst_norm[i]
        A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])

    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)

    # Denormalize H
    H = np.dot(np.linalg.inv(T_src), np.dot(H, T_dst))
    H /= H[-1][-1]
    return H

image_names = ["0000.png","0001.png","0002.png","0003.png","0004.png","0005.png","0006.png","0007.png","0008.png","0009.png","0010.png"]
images = []
for image_name in image_names:
    image_path = r"images\images\\"+image_name
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray_image = gray_image.astype("float32")
    images.append(gray_image)

#特征提取
key_des = []
for i in range(11):
    keypoints,descriptors = feature_extraction.extract_features(images[i])
    key_des.append((keypoints,descriptors))

matching_list= []
for i in range(10):
    keypoints1,descriptors1 = key_des[i:i+2][0]
    keypoints2,descriptors2 = key_des[i:i+2][1]
    matching_list.append(list(feature_matching.match_features(keypoints1,descriptors1,keypoints2,descriptors2)))

with open("camera_intrinsic.txt",'r') as File:
    lines = File.readlines()

camera_intrinsic = []
for line in lines:
    row = [float(x) for x in line.split()]
    camera_intrinsic.append(row)

camera_intrinsic = np.array(camera_intrinsic)    

R1,t1 = initial_recon.initial_recon(matching_list[0][0],matching_list[0][1],camera_intrinsic)
R1,t1
T1 = np.hstack((camera_intrinsic,np.zeros((camera_intrinsic.shape[0],1))))
T2 = np.hstack((R1,t1))

for i in matching_list:
    dst_pts, src_pts = i[0],i[1]













# def normalize_points(points):
#     homo_points = np.hstack((points, np.ones((points.shape[0], 1))))
#     mean = np.mean(homo_points[:, :-1], axis=0)
#     scale = np.sqrt(2) / np.std(homo_points[:, :-1], axis=0)
#     T = np.diag([scale[0], scale[1], 1])
#     T[:-1, -1] = -mean * scale
#     norm_points = np.dot(T, homo_points.T).T
#     return norm_points, T
# def DLT(src_norm, dst_norm,T_src,T_dst):


#     A = []
#     for i in range(src_norm.shape[0]):
#         x, y, _ = src_norm[i]
#         xp, yp, _ = dst_norm[i]
#         A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
#         A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])

#     A = np.array(A)
#     U, S, Vt = np.linalg.svd(A)
#     H = Vt[-1].reshape(3, 3)

#     # Denormalize H
#     H = np.dot(np.linalg.inv(T_src), np.dot(H, T_dst))
#     H /= H[-1][-1]
#     return H
