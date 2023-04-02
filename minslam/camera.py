import numpy as np
from spatialmath import SE3
import cv2
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import gtsam

class PinholeCamera():
    def __init__(self, params):
        # Parameters
        self.params = params
        self.camera_matrix = [float(x) for x in params['dataset']['camera_matrix'].split(' ')]
        self.body_T_cam = self._load_body_T_cam(params['dataset']['body_T_cam'])
    
    def _load_body_T_cam(self, data_str):
        '''
        @param data_str: camera pose in string format
        @return: camera pose in the body frame
        '''
        data = [float(x) for x in data_str.split(' ')]
        data = np.array(data).reshape(4,3)
        body_T_cam = np.zeros((4,4))
        body_T_cam[:3, :3] = data[:3, :]
        body_T_cam[:3, 3] = data[3, :].transpose()
        body_T_cam[3, 3] = 1
        return SE3(body_T_cam)
    
    def project(self, point, body_pose):
        '''
        @param point: 3D point in the world frame
        @param body_pose: pose of the body in world frame
        @return: 2D point in the image plane
        '''
        fx, fy, cx, cy = self.camera_matrix
        point0 = body_pose.inv() * point
        point1 = self.body_T_cam.inv() * point0
        d = point1[2]
        u = point1[0]*fx/d + cx
        v = point1[1]*fy/d + cy
        return u, v, d
    
    def back_project(self, u, v, depth, body_pose):
        '''
        @param u: x coordinate of the pixel
        @param v: y coordinate of the pixel
        @param depth: depth of the pixel
        @param body_pose: pose of the body in world frame
        @return: 3D point in the world frame
        '''
        fx, fy, cx, cy = self.camera_matrix
        point0 = np.array([
                    (u-cx)*depth/fx,
                    (v-cy)*depth/fy,
                    depth
                ])
        point1 = self.body_T_cam * point0
        point2 = body_pose * point1
        return point2
    
    def back_project2(self, u, v, depth):
        '''
        back_project without transformation
        '''
        fx, fy, cx, cy = self.camera_matrix
        return np.array([
                    (u-cx)*depth/fx,
                    (v-cy)*depth/fy,
                    depth
                ])