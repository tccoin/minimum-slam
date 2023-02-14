import numpy as np
from spatialmath import *
import cv2
import matplotlib.pyplot as plt


class Frontend():
    def __init__(self, params):
        self.params = params
        self.curr_odom_pose = None
        self.last_odom_pose = None
        self.curr_color = None
        self.last_color = None
        self.curr_gray = None
        self.last_gray = None
        self.last_depth = None
        self.curr_depth = None
        self.last_points = None
        self.curr_points = None
        self.last_keypoints = None
        self.curr_keypoints = None
        self.last_matches = None
        self.curr_matches = None
        self.last_descriptors = None
        self.curr_descriptors = None
        self.frame_id = -1

    def keyframe_selection(self, odom_pose) -> bool:
        '''
        @param odom_pose: current odometry pose
        @return: True if a keyframe is selected
        '''
        is_keyframe = False
        if self.curr_odom_pose is None:
            # First frame
            is_keyframe = True
        else:
            # Determine if the motion is large enough
            rel_pose = self.curr_odom_pose.inv() * odom_pose
            w_trans = self.params['frontend']['keyframe']['trans_weight']
            w_rot = self.params['frontend']['keyframe']['rot_weight']
            weight = np.array([w_trans]*3+[w_rot]*3)
            diff = np.linalg.norm(rel_pose.log(True) * weight)
            threshold = self.params['frontend']['keyframe']['threshold']
            is_keyframe = diff > threshold
        return is_keyframe

    def add_keyframe(self, odom_pose, color, depth) -> None:
        '''
        @param odom_pose: current pose
        @param color: current color image
        @param depth: current depth image
        '''
        self.frame_id += 1
        self.last_odom_pose = self.curr_odom_pose
        self.curr_odom_pose = odom_pose
        self.last_color = self.curr_color
        self.last_gray = self.curr_gray
        self.last_depth = self.curr_depth
        self.curr_color = color
        self.curr_gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        self.curr_depth = depth
        self.last_points = self.curr_points
        self.last_keypoints = self.curr_keypoints
        self.last_matches = self.curr_matches
        self.last_descriptors = self.curr_descriptors
        # empty the results
        self.curr_points = None
        self.curr_keypoints = None
        self.curr_matches = None
        self.curr_descriptors = None

    def extract_features(self, detector_name=None, sample_method=None) -> None:
        # select detector
        n = self.params['frontend']['feature']['number']
        if detector_name is None:
            detector_name = self.params['frontend']['feature']['detector']
        if detector_name == 'sift':
            detector = cv2.SIFT_create(n)
        elif detector_name == 'orb':
            detector = cv2.ORB_create(n)
        elif detector_name == 'akaze':
            detector = cv2.AKAZE_create()

        # todo: add sample methods
        if sample_method is None:
            sample_method = self.params['frontend']['feature']['sample']
        if sample_method == 'vanilla':
            self.curr_keypoints, self.curr_descriptors = detector.detectAndCompute(
                self.curr_color, None)
        else:
            raise NotImplementedError
        self.curr_points = np.array(
            [x.pt for x in self.curr_keypoints], dtype=np.float32)

    def match_features(self, matcher_name=None, sample_method=None) -> None:
        if self.frame_id == 0:
            return
        # select matcher
        if matcher_name is None:
            matcher_name = self.params['frontend']['match']['matcher']
        if matcher_name == 'bruteforce':
            cross_check = self.params['frontend']['match']['cross_check']
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=cross_check)
            self.curr_matches = matcher.match(
                self.last_descriptors, self.curr_descriptors)
        elif matcher_name == 'opticalflow':
            self.curr_points, status, err = cv2.calcOpticalFlowPyrLK(self.last_gray, self.curr_gray,
                                                                     self.last_points, None, winSize=(21, 21), maxLevel=3)
        else:
            raise NotImplementedError

    def eliminate_outliers(self):
        pass

    def plot_features(self):
        canvas = np.array(self.curr_color)
        for point in self.curr_points:
            cv2.circle(canvas, (int(point[0]), int(
                point[1])), 4, (255, 0, 0), 1)
        plt.imshow(canvas[:, :, ::-1])

    def plot_matches(self):
        if self.frame_id == 0:
            self.plot_features()
        else:
            canvas = np.concatenate([self.last_color, self.curr_color], axis=1)
            for match in self.curr_matches:
                pt1 = self.last_points[match.queryIdx]
                pt2 = self.curr_points[match.trainIdx] + \
                    np.array([self.last_color.shape[1], 0])
                cv2.circle(canvas, (int(pt1[0]), int(
                    pt1[1])), 4, (255, 0, 0), 1)
                cv2.circle(canvas, (int(pt2[0]), int(
                    pt2[1])), 4, (255, 0, 0), 1)
                cv2.line(canvas, (int(pt1[0]), int(pt1[1])), (int(
                    pt2[0]), int(pt2[1])), (255, 0, 0), 1)
            plt.imshow(canvas[:, :, ::-1])


if __name__ == '__main__':
    frontend = Frontend('./params/tartanair.yaml')
    new_pose = SE3()
    frontend.keyframe_selection(new_pose)
    new_pose = SE3.Rx(0.5)
    frontend.keyframe_selection(new_pose)
