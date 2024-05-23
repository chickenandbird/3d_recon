import numpy as np
from scipy.sparse import lil_matrix
import cv2
import matplotlib.pyplot as plt

def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.
    
    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    cmat = np.matrix([[2759.48,0,1520.69],[0,2764.16,1006.81],[0,0,1]])
    points_proj= np.array(points_proj @ cmat.T)
    
    for i in range(points_proj.shape[0]):
        points_proj[i][0] /= points_proj[i][2]
        points_proj[i][1] /= points_proj[i][2]
    return points_proj[:,:2]

def fun_2(params, n_cameras, n_points, camera_indices, point_indices, fun_points_2d,maxmin = 2000):
    """Compute residuals.
    
    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    fun_points_3d = params[n_cameras * 6:].reshape((n_points, 3))
    points_proj = project(fun_points_3d[point_indices], camera_params[camera_indices])
    respp = (points_proj - fun_points_2d[point_indices].squeeze()).ravel()
    return respp

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 6 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1

    return A
def get_params(list1,d3_po_all,d2_po_all):
    cameras = []
    parm_points_3d = []
    parm_points_2d = []
    point_indices = []
    camera_indices = []
    cnt = 0
    for i in range(1,len(list1)):
        rot_vec = np.zeros((3,))
        cv2.Rodrigues(list1[i][:3,:3], rot_vec)
        x_cam = np.zeros((6))
        x_cam[:3] = rot_vec
        x_cam[3:] = list1[i][:3,3]
        cameras.append(x_cam)
    for i in range(10):
        for j in range(len(d3_po_all[i])):
            camera_indices.append(i)
            parm_points_2d.append(d2_po_all[i][j])
            parm_points_3d.append(d3_po_all[i][j])
            point_indices.append(cnt)
            cnt += 1
    return np.array(cameras),np.squeeze(np.array(parm_points_3d)),np.array(parm_points_2d),np.array(camera_indices),np.array(point_indices)
def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 6 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1

    return A














# def reprojection_error(params, num_cameras, num_points):
#     """
#     定义重投影误差函数
#     """
#     # 将优化参数拆分为相机参数和三维点坐标
#     camera_params = params[:num_cameras * 12].reshape((num_cameras, 3, 4))
#     points_3d_opt = params[num_cameras*12:num_cameras*12 + num_points*3].reshape((num_points, 3))
#     points_2d_opt = params[num_cameras*12+num_points*3 : num_cameras*12+num_points*5].reshape((num_points, 2))
#     K = params[num_cameras*12+num_points*5 :].reshape((3, 3))


#     # 计算重投影误差
#     errors = []
#     for i in range(num_cameras):
#         R = camera_params[i][:3, :3]
#         t = camera_params[i][:3, 3]

#         proj = np.dot(K, np.hstack((R, t[:, np.newaxis])))

#         proj_3d = np.dot(proj, np.vstack((points_3d_opt.T, np.ones(num_points))))

#         proj_2d = proj_3d[:2, :] / proj_3d[2, :]
#         proj_2d = proj_2d.T
#         errors.append((proj_2d - points_2d_opt[i]).ravel())
#     return np.concatenate(errors)

# def bundle_adjustment(camera0,points_3d, points_2d, num_cameras, num_points, K):
#     """
#     Bundle Adjustment 主函数
#     """
#     # 初始化相机参数和三维点坐标
#     params = np.zeros((num_cameras * 12 + num_points * 5  + 9))

#     # 设置优化参数的初始值
#     params[:12] = camera0[:3].flatten()

#     params[num_cameras*12:num_cameras*12+num_points*3] = points_3d.ravel()

#     params[num_cameras*12+num_points*3 : num_cameras*12+num_points*5] = points_2d.ravel()

#     params[num_cameras*12+num_points*5 :] = K.flatten()
#     # 执行优化
#     res = least_squares(reprojection_error, params, args=(num_cameras, num_points))

#     # 从优化结果中获取相机参数和三维点坐标
#     camera_params_opt = res.x[:num_cameras * 12].reshape((num_cameras, 3, 4))
#     points_3d_opt = res.x[num_cameras*12:num_cameras*12 + num_points*3].reshape((num_points, 3))

#     return camera_params_opt, points_3d_opt




















# from scipy.spatial.transform import Rotation
# from scipy.linalg import expm
# from sophus.sophus_pybind import SE3
# def bundle_adjustment(camera0,points_3d, points_2d,K):
#     iter = 10
#     cost,last_cost = 0,9999999999999999
#     fx,fy,cx,cy = K[0][0],K[1][1],K[0][2],K[1][2]
#     R = camera0[:3, :3]
#     t = camera0[:3, 3]   
#     # pose = np.hstack((R,t[:, np.newaxis]))
#     pose = camera0
#     for i in range(iter):
#         cost = 0
#         H = np.zeros((6,6))
#         b = np.zeros((6,1))
#         for j,(point_3d,point_2d) in enumerate(zip(points_3d,points_2d)):
#             R = pose[:3, :3]
#             t = pose[:3, 3]   
#             D = np.hstack((R, t[:, np.newaxis]))
#             L = np.vstack((point_3d.T, np.ones(1)))
#             pc = np.dot(D, L)
#             inv_z = (1/pc[2])[0]
#             inv_z2 = inv_z*inv_z
#             proj3d = np.dot(K,pc)
#             e = point_2d.reshape((2,1))-proj3d[:2]
#             cost += np.sum(np.square(e))
#             J = np.array([[-fx*inv_z, 0, fx*pc[0][0]*inv_z2, fx*pc[0][0]*pc[1][0]*inv_z2, -fx - fx * pc[0][0] * pc[0][0] * inv_z2,fx * pc[1][0] * inv_z],
#                 [0,-fy * inv_z, fy*pc[1][0]*inv_z, fy+fy*pc[1][0]*pc[1][0]*inv_z2, -fy * pc[0][0] * pc[1][0] * inv_z2,-fy * pc[0][0] * inv_z]])
#             H += np.dot(J.transpose(),J)
#             b -= np.dot(J.transpose(),e)

#         if cost>last_cost:
#             break

#         dx = np.linalg.solve(H, b)

#         delta_pose = expm(dx)

#         pose = delta_pose *(np.eye(3) + pose)

#         last_cost = cost

#         print(cost)

#     return pose
            
# camera_params_opt_all = []
# camera_params_opt_all.append(np.vstack((np.hstack((np.eye(3),np.zeros((3,1)))),np.array([0,0,0,1]))))
# for i in range(10):
#     camera_params_opt = bundle_adjustment(list0[i+1],points_3d_new_all[i],src_pts_new_all[i],camera_intrinsic)
    
#     camera_params_opt_all.append(camera_params_opt[0])
#     # print("Optimized camera parameters:")
#     # print(camera_params_opt)
#     # print("Optimized 3D points:")
#     # print(points_3d_opt)


# with open ("521030910127.txt","w") as file:
#     for i in camera_params_opt_all:
#         np.savetxt(file, i.reshape(1,-1))  # 写入第一个矩阵
#         file.write('\n')  # 换行