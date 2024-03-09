"ICP: https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html"


from ...utils import settings

from klampt.math import se3
from klampt.io import numpy_convert
from klampt.model.sensing import image_to_points

import open3d as o3d

import cv2
import os
import numpy as np
import math
import yaml

class Zed:
    def __init__(self):
        with open(settings.get('calibration.zed_intrinsics'), 'r') as file:
            config = yaml.load(file, yaml.SafeLoader)
            self.K = np.array(config['K']).reshape(3,3)
            self.fx, self.fy = self.K[0,0], self.K[1,1]
            self.cx, self.cy = self.K[0,2], self.K[1,2]
            self.h = config['height']
            self.w = config['width']

        self.xFov = 2 * np.arctan(self.w/ (2 * self.fx))
        self.yFov = 2 * np.arctan(self.h/ (2 * self.fy))

        transpose_flatten = lambda l : np.array(l).transpose().flatten().tolist()
        R = transpose_flatten([[0,0,1], [-1,0,0], [0,-1,0]])
        self.T = (R, [0,0,0])

    def print(self):
        print('----Zed----')
        print('f: ', self.fx, self.fy, ' | c: ', self.cx, self.cy)
        print('h:', self.h, " | w: ", self.w)
        print('FOV:', math.degrees(self.xFov), math.degrees(self.yFov))
        print('T:', self.T)

    def get_point_cloud(self, folder, idx):
        depth_data_path = os.path.join(folder, 'depth{}.tif').format(idx)
        depth_data = cv2.imread(depth_data_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

        points = image_to_points(depth_data, None, self.xFov, self.yFov, depth_scale=4000.0/0xffff)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd


class Lidar:
    def __init__(self):
        self.T = se3.identity()

    def print(self):
        print('Lidar params')
        print('T:', self.T)

    def get_point_cloud(self, folder, idx):
        lidar_data_path = os.path.join(folder, 'lidar{}.npz').format(idx)
        lidar_data = np.load(lidar_data_path)['arr_0']

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(lidar_data)
        return pcd


def register_points(source, target, T):
    d_max = 0.05
    T_init = np.identity(4)
    T_init[:3,:3] = np.array(T[0]).reshape((3,3))

    reg = o3d.pipelines.registration.registration_icp(source, target, d_max, T_init, \
                                                      o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print(reg)

    return reg.transformation


def run(folder, idx):
    zed = Zed()
    zed.print()
    zed_pcd = zed.get_point_cloud(folder, idx)
    print(zed_pcd)

    lidar = Lidar()
    lidar.print()
    lidar_pcd = lidar.get_point_cloud(folder, idx)
    print(lidar_pcd)

    T_lidar_zed = register_points(lidar_pcd, zed_pcd, zed.T)
    print('T_velodyne^zed:')
    print(T_lidar_zed)


path = input("enter data path")
index = int(input("enter data index"))
print(path)
run(path, index)