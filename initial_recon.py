# use epipolar geometry to find the fundamental matrix and the essential matrix
# use the essential matrix to find the relative pose between two cameras
# use the relative pose to triangulate 3D points

import numpy as np
import cv2
import feature_extraction
import feature_matching
import open3d as o3d
def show3dpts(pts3dall):
    points=o3d.utility.Vector3dVector(pts3dall)
    pcd=o3d.geometry.PointCloud()
    pcd.points = points
   
    vis = o3d.visualization.Visualizer()
    vis.create_window()	
    render_option: o3d.visualization.RenderOption = vis.get_render_option()	
    render_option.background_color = np.array([0, 0, 0])	
    render_option.point_size = 2.0	
    vis.add_geometry(pcd)	
    vis.run()
    vis.destroy_window()

#利用对极几何进行场景初始化，求出前两张图的基本矩阵和本质矩阵
def initial_recon_func(p1,p2,camera_intrinsic):
    essencial_martrix,_ = cv2.findEssentialMat(p1,p2,camera_intrinsic)
    _,R,t,_ = cv2.recoverPose(essencial_martrix,p1,p2,camera_intrinsic)
    return R,t



if __name__ == "__main__":
    image_names = ["0000.png","0001.png","0002.png","0003.png","0004.png","0005.png","0006.png","0007.png","0008.png","0009.png","0010.png"]
    images = []
    for image_name in image_names:
        image_path = r"images\\images\\"+image_name
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
        keypoints1,descriptors1 = key_des[0]
        keypoints2,descriptors2 = key_des[i+1]
        matching_list.append(list(feature_matching.match_features(keypoints1,descriptors1,keypoints2,descriptors2)))

    with open("camera_intrinsic.txt",'r') as File:
        lines = File.readlines()
    camera_intrinsic = []
    for line in lines:
        row = [float(x) for x in line.split()]
        camera_intrinsic.append(row)

    camera_intrinsic = np.array(camera_intrinsic)

    points_3d_all = []
    list0 = []
    list0.append(np.vstack((np.hstack((np.eye(3),np.zeros((3,1)))),np.array([0,0,0,1]))))
    for i in range(10):
        cur_dst_pts = matching_list[i][1]
        cur_src_pts = matching_list[i][2]
        R,t = initial_recon_func(cur_dst_pts,cur_src_pts,camera_intrinsic)#在这里我把src提到了前面作为points1，也就是0001.png，最终求出了使得x1 = Rx2+t的结果
        cur_T1 = np.hstack((camera_intrinsic,np.zeros((camera_intrinsic.shape[0],1))))
        cur_T2 = np.hstack((np.dot(camera_intrinsic,R),np.dot(camera_intrinsic,t)))
        cur_points_4d_homogeneous = cv2.triangulatePoints(cur_T1,cur_T2,cur_dst_pts,cur_src_pts)
        cur_points_3d = cv2.convertPointsFromHomogeneous(cur_points_4d_homogeneous.T)    
        points_3d_all.append(cur_points_3d)
        M = np.hstack((R,t))
        M = np.vstack((M,np.array([0,0,0,1])))

        list0.append(M)
    with open ("epipolar.txt","w") as file:
        for i in list0:
            np.savetxt(file, i.reshape(1,-1))  # 写入第一个矩阵
            file.write('\n')  # 换行
    def delete_too_big(n):
        for i in n:
            if abs(i)>10:
                return False
        return True
    listk = []
    for i in points_3d_all:
        listk.extend([i.squeeze()[j] for j in range(i.shape[0])])
    print(len(listk))
    listk = filter(delete_too_big,listk)
    listk = list(listk)
    print(len(listk))
    show3dpts(listk)
