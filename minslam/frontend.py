import numpy as np
from spatialmath import *
import cv2
import matplotlib.pyplot as plt

class FrontendKeyframe():
    def __init__(self) -> None:
        self.frame_id = -1
        self.odom_pose = None
        self.color = []
        self.gray = []
        self.depth = []
        self.points = []
        self.keypoints = []
        self.descriptors = []
        self.matches = []
        self.global_id = []


class Frontend():
    def __init__(self, params):
        # Parameters
        self.params = params

        # All states
        self.last_frame = FrontendKeyframe()
        self.curr_frame = FrontendKeyframe()
        self.frame_id = -1

    def keyframe_selection(self, odom_pose) -> bool:
        '''
        @param odom_pose: current odometry pose
        @return: True if a keyframe is selected
        '''
        is_keyframe = False
        if self.curr_frame.odom_pose is None:
            # First frame
            is_keyframe = True
        else:
            # Determine if the motion is large enough
            rel_pose = self.curr_frame.odom_pose.inv() * odom_pose
            w_trans = self.params['frontend']['keyframe']['trans_weight']
            w_rot = self.params['frontend']['keyframe']['rot_weight']
            weight = np.array([w_trans]*3+[w_rot]*3)
            if rel_pose == np.eye(4):
                diff = 0
            else:
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
        self.last_frame = self.curr_frame
        self.curr_frame = FrontendKeyframe()
        self.curr_frame.odom_pose = odom_pose
        self.curr_frame.color = color
        self.curr_frame.gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        self.curr_frame.depth = depth

    def extract_features(self, n=None, detector_name=None, sample_method=None, append_mode=False) -> None:
        # select detector
        if n is None:
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
            keypoints, descriptors = detector.detectAndCompute(
                self.curr_frame.color, None)
        else:
            raise NotImplementedError
        if not append_mode:
            self.curr_frame.keypoints = keypoints
            self.curr_frame.descriptors = descriptors
            self.curr_frame.points = [x.pt for x in self.curr_frame.keypoints]
        else:
            self.curr_frame.keypoints += keypoints
            self.curr_frame.points += [x.pt for x in keypoints]

    def match_features(self, matcher_name=None, sample_method=None) -> None:
        if self.frame_id == 0:
            return
        # select matcher
        if matcher_name is None:
            matcher_name = self.params['frontend']['match']['matcher']
        if matcher_name == 'bruteforce':
            cross_check = self.params['frontend']['match']['cross_check']
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=cross_check)
            self.curr_frame.matches = matcher.match(self.last_frame.descriptors, self.curr_frame.descriptors)
        elif matcher_name == 'opticalflow':
            curr_points_all, status, err = cv2.calcOpticalFlowPyrLK(
                self.last_frame.gray, self.curr_frame.gray,
                np.array(self.last_frame.points, dtype=np.float32), None, winSize=(21, 21), maxLevel=3)
            self.curr_frame.points = []
            self.curr_frame.matches = []
            for i, (point, is_inlier) in enumerate(zip(curr_points_all, status)):
                if is_inlier:
                    query_idx = i
                    train_idx = len(self.curr_frame.points)
                    self.curr_frame.points.append(point)
                    self.curr_frame.matches.append(cv2.DMatch(query_idx, train_idx, 0))
        else:
            raise NotImplementedError

    def eliminate_outliers(self):
        if len(self.curr_frame.matches)<8:
            return
        last_points = np.array([self.last_frame.points[match.queryIdx] for match in self.curr_frame.matches])
        curr_points = np.array([self.curr_frame.points[match.trainIdx] for match in self.curr_frame.matches])
        retval, mask = cv2.findFundamentalMat(last_points, curr_points, cv2.FM_RANSAC, 3, 0.99, None)
        
        matches = []
        for i, is_inlier in enumerate(mask):
            if is_inlier:
                matches.append(self.curr_frame.matches[i])
        self.curr_frame.matches = matches

    def plot_features(self):
        canvas = np.array(self.curr_frame.color)
        for point in self.curr_frame.points:
            cv2.circle(canvas, (int(point[0]), int(
                point[1])), 4, (255, 0, 0), 1)
        plt.imshow(canvas[:, :, ::-1])

    def plot_matches(self, with_global_id=False):
        if self.frame_id == 0:
            self.plot_features()
        else:
            canvas = np.concatenate([self.last_frame.color, self.curr_frame.color], axis=1)
            for match in self.curr_frame.matches:
                pt1 = [int(x) for x in self.last_frame.points[match.queryIdx]]
                pt2 = [int(x) for x in self.curr_frame.points[match.trainIdx]]
                pt2[0] += self.last_frame.color.shape[1]
                cv2.circle(canvas, pt1, 4, (255, 0, 0), 1)
                cv2.circle(canvas, pt2, 4, (255, 0, 0), 1)
                cv2.line(canvas, (int(pt1[0]), int(pt1[1])), (int(
                    pt2[0]), int(pt2[1])), (255, 0, 0), 1)
                if with_global_id:
                    pt1[0] += 10
                    pt1[1] += 10
                    pt1[0] += 10
                    pt2[1] += 10
                    cv2.putText(canvas, str(self.last_frame.global_id[match.queryIdx]), pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255))
                    cv2.putText(canvas, str(self.curr_frame.global_id[match.trainIdx]), pt2, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255))
            plt.imshow(canvas[:, :, ::-1])
    
    def assign_global_id(self):
        if self.frame_id == 0:
            self.curr_frame.global_id = [i for i in range(len(self.curr_frame.points))]
        else:
            self.curr_frame.global_id = [-1 for i in range(len(self.curr_frame.points))]
            for match in self.curr_frame.matches:
                self.curr_frame.global_id[match.trainIdx] = self.last_frame.global_id[match.queryIdx]


if __name__ == '__main__':
    frontend = Frontend('./params/tartanair.yaml')
    new_pose = SE3()
    frontend.keyframe_selection(new_pose)
    new_pose = SE3.Rx(0.5)
    frontend.keyframe_selection(new_pose)
