#!/usr/bin/env python

#plug into SLAM algo (runs with just 2 terminals)

# pip install numpy spatialmath-python opencv-python matplotlib gtsam evo plotly
# pip install -U numpy

# import roslib
# roslib.load_manifest('bagloader')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
# from bagpy import bagreader
import message_filters
from geometry_msgs.msg import PoseWithCovarianceStamped
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os

#MinSLAM imports
# frontend
import numpy as np
from spatialmath import *
import cv2
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# backend
import gtsam
from gtsam.symbol_shorthand import L, X

import numpy as np
from spatialmath import SE3
import cv2
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import gtsam
from gtsam.symbol_shorthand import L, X
from minslam.camera import PinholeCamera
# import matplotlib 
# matplotlib.use('agg')

# our slam implementation
from minslam.data_loader import TartanAirLoader, plot_trajectory
from minslam.frontend import Frontend 
from minslam.params import Params
from minslam.backend import Backend
from minslam.camera import PinholeCamera

import time



class MessageListener:
    def __init__(self):
        # self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        # self.depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)
        self.image_sub = message_filters.Subscriber("/camera/color/image_raw",Image)
        self.depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw',Image)
        self.odom_sub = message_filters.Subscriber('/cheetah/inekf_estimation/pose',PoseWithCovarianceStamped)
        self.posecache = message_filters.Cache(self.odom_sub, 1) #set cache size to 1
        # self.imagecache = message_filters.Cache(self.image_sub, 1) #set cache size to 1
        # self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub], 1400, 0.025)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub], 1400, 0.08)
        self.ts.registerCallback(self.imde_callback)
        self.bridge = CvBridge()
        self.params = Params('~/ws/ros_ws/src/minimum-slam/params/cheetah.yaml') #change this later
        self.frontend = Frontend(self.params)
        self.backend = Backend(self.params) 
        self.estimated_traj= np.empty((0, 3)) 

     
     #complet04: there are 174 total images messages and 121 total depth messages, so the max buffer should be 
     #less than or equal to 174. the avg time btwn image and depth messages is 0.2s, so pick 0.5 to start

     #outdoor: there are 1333 total images messages (at 284Hz- actually it's 24Hz)
     #and 424 total depth messages (at 7.5Hz), so the max buffer should be less than or equal 
     #to 1333. the avg time btwn image and depth messages is 0.04s, so pick 0.2 to start
    
     #indoor: there are 962 total images messages (at 27Hz) and 866 total depth messages (at 69Hz).
     # the avg time btwn image and depth messages is 0.04s, so pick 0.2 to start
     
     #to do: check new time diff pattern; check if image
     #and depth time stamps are close to eachother and image is not empty before continuing
     #make sure depth image is correct
     #figure out how to improve feature matching

        self.file_path_1 = 'time_diff.txt'
        with open(self.file_path_1, 'w') as f:
            f.write("[depth image depthvsim]\n")

        self.file_path_2 = 'keyframes.txt'
        with open(self.file_path_2, 'w') as f:
            f.write("keyframe:\n")

        self.file_path_3 = 'est_traj.txt'
        with open(self.file_path_3, 'w') as f:
            f.write("est_traj:\n")
            
        # self.file_path_4 = 'image.txt'
        # with open(self.file_path_4, 'w') as f:
        #     f.write("image:\n")

    # def depth_callback(self, depth_msg):
        # with open(self.file_path_3, 'a') as f:
        #         depth_timestamp = depth_msg.header.stamp.to_sec()
        #         f.write(f"{depth_timestamp}\n")
        
    def imde_callback(self, image, depth):
 
        # Extract the timestamp from the depth message
        if image is not None and hasattr(image, 'header'):
            depth_timestamp = depth.header.stamp.to_sec()
            image_timestamp = image.header.stamp.to_sec()
            time_diff = depth_timestamp - image_timestamp

            if time_diff < 0.09:

                with open(self.file_path_1, 'a') as f:
                    f.write(f"{depth_timestamp}\t{image_timestamp}\t{time_diff}\n")

                cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
                cv_depth = self.bridge.imgmsg_to_cv2(depth, desired_encoding='passthrough')

                pose_msg = self.posecache.getElemBeforeTime(self.posecache.getLatestTime()) #since the cache size is 1, wouldn't this be the same as saying self.cache[0]
                pose_se3 = self.pose_with_covariance_to_se3(pose_msg)
                # cv2.imshow("RGB Image", cv_image)
                # cv2.imshow("Depth Image", cv_depth)
                # cv2.waitKey(200)

                if not self.frontend.keyframe_selection(pose_se3):
                    return
                else:
                    frontend_frame = self.run_frontend_once(cv_image,cv_depth,pose_se3)
                    measurements = []
                    frame_id = frontend_frame.frame_id
                    count = 0
                    for landmark in frontend_frame.landmarks:
                        global_id = landmark.global_id
                        measurement = landmark.measurements[frame_id] # u, v, depth
                        # if not landmark.is_outlier_gt(frontend, 0.01):
                        count += 1
                        measurements.append((global_id, *measurement))
                    print(f'add {count} measurements to backend\n')
                    # add measurements to backend
                    self.backend.add_keyframe(frame_id, pose_se3, measurements)
                    # optimize the backend
                    self.backend.optimize(optimizer='ISAM2')
                    self.backend_estimate = self.backend.current_estimate
                    est = gtsam.utilities.extractPose3(self.backend_estimate)[:, -3:]
                    self.estimated_traj = np.concatenate((self.estimated_traj, est), axis=0)
                    with open(self.file_path_3, 'a') as f:
                        f.write(f"{self.estimated_traj}\n")

            else:
                print('Error: Image and Depth Data not aligned')
                return

    # def image_callback(self, image):
    #     with open(self.file_path_4, 'a') as f:
    #         image_timestamp = image.header.stamp.to_sec()
    #         f.write(f"{image_timestamp}\n")

    def pose_with_covariance_to_se3(self, pose_msg):
        # Extract pose information
        position = pose_msg.pose.pose.position
        orientation = pose_msg.pose.pose.orientation

        # Convert position to a numpy array
        position_array = np.array([position.x, position.y, position.z])

        # Convert orientation to a numpy array
        orientation_array = np.array([orientation.x, orientation.y, orientation.z, orientation.w])

        # Convert quaternion to rotation matrix
        rotation_matrix_2 = self.quaternion_to_rotation_matrix2(orientation_array)
        # print(orientation_array)
        # print(rotation_matrix_2)

        se3_matrix_2 = np.eye(4)
        se3_matrix_2[:3, :3] = rotation_matrix_2
        se3_matrix_2[:3, 3] = position_array
        se3_obj = SE3(se3_matrix_2)

        T = np.array([[0,-1,0,0],
                        [0,0,-1,0],
                        [1,0,0,0],
                        [0,0,0,1]], dtype=np.float32) 
        T_inv = np.linalg.inv(T)
        se3_obj = SE3(T.dot(se3_matrix_2).dot(T_inv))
      
        print(se3_obj)
        print(SO3(se3_obj.R).eulervec()/np.pi*180)

        return se3_obj
    
    def quaternion_to_rotation_matrix2(self, quaternion):
        # source: https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/
        # Extract the values from Q
        Q=quaternion
        q0 = Q[3]
        q1 = Q[0]
        q2 = Q[1]
        q3 = Q[2]
        
        # First row of the rotation matrix
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)
        
        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)
        
        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1
        
        # 3x3 rotation matrix
        rot_matrix = np.array([[r00, r01, r02],
                            [r10, r11, r12],
                            [r20, r21, r22]])
                                
        return rot_matrix

    def run_frontend_once(self, image, depth, pose):          
        self.frontend.add_keyframe(pose, image, depth)
        print(f'--- Added keyframe {self.frontend.frame_id}')
        with open(self.file_path_2, 'a') as f:
           f.write(f"{self.frontend.frame_id}\n")
        more_points_n = self.params['frontend']['feature']['number']
        self.frontend.extract_features(more_points_n, append_mode=False)
        print('extracting features:', len(self.frontend.curr_frame.keypoints), f'(expected {more_points_n})')
        if self.frontend.frame_id > 0:
            self.frontend.match_features()
            print('matching features:', len(self.frontend.curr_frame.matches))
            self.frontend.eliminate_outliers()
        self.frontend.assign_global_id()

        # fig=self.frontend.plot_matches()
        # fig.show()
        # save_filename = f'keyframe_{self.frontend.frame_id}.png'
        # fig.savefig(save_filename)
        # plt.close(fig) 
        return self.frontend.curr_frame

def main(args):
    # Change directory to the bag file location
    bag_file = os.path.expanduser('~/ws/ext_data/cheetah_outdoor.bag')
    
    # Play ROS bag on loop
    subprocess.Popen(['rosbag', 'play','-q','-r','0.5', bag_file])

    rospy.init_node('message_listener', anonymous=True)
    listener = MessageListener()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

    fig = go.Figure()
    plot_trajectory(listener.estimated_traj, 'estimated', fig)

if __name__ == '__main__':
    main(sys.argv)

