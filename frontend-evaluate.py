import os
import cv2
import numpy as np
from numpy.linalg import inv, norm

def quaternion_to_rotation_matrix(Q):
    # from https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
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

class Landmark():
    def __init__(self, landmark_id):
        self.landmark_id = landmark_id
        self.measurements = {}
        self.depths = {}
        self.gt = {}
    def add_measurement(self, frame_id, u, v):
        self.measurements[frame_id] = [u, v]
    def add_depth(self, frame_id, d):
        self.depths[frame_id] = d
    def add_gt(self, frame_id, u, v):
        self.gt[frame_id] = [u, v]

class Frame():
    def __init__(self):
        self.observed_landmark_id = []
    def add_landmark_id(self, landmark_id):
        self.observed_landmark_id.append(landmark_id)
    def add_pose(self, data):
        rotation = np.array([float(x) for x in data[0:9]]).reshape((3,3))
        translation = np.array([float(x) for x in data[9:12]]).reshape((3,1))
        self.rotation = rotation
        self.translation = translation

class FrontendEvaluate():
    def __init__(self, dataset_type, dataset_folder):
        self.dataset_type = dataset_type
        self.dataset_folder = dataset_folder

    def load_frontend_output(self, file_path):
        self.landmarks = {}
        self.frames = {}
        self.format = '0'
        self.camera = []
        with open(file_path, 'r') as file:
            current_frame_id = 0
            for line in file:
                data = line[:-1].split(' ')
                if data[0]=='FEATURE':
                    landmark_id = int(data[1])
                    if not landmark_id in self.landmarks:
                        self.landmarks[landmark_id] = Landmark(landmark_id)
                        self.frames[current_frame_id].add_landmark_id(landmark_id)
                    self.landmarks[landmark_id].add_measurement(current_frame_id, float(data[2]), float(data[3]))
                elif data[0]=='FRAME':
                    current_frame_id = int(data[1])
                    self.frames[current_frame_id]=Frame()
                elif data[0]=='FORMAT':
                    self.format = data[1]
                    print('Format version: ', self.format)
                    if self.format != '0':
                        print('Wrong format version! Program will quit.')
                        quit()
                elif data[0]=='CAMERA':
                    self.camera = [int(x) for x in data[1:]]
                    print('Loading camera: ', self.camera)
                elif data[0]=='CALIBRATION':
                    self.cameraPose = Frame() # to load the pose
                    self.cameraPose.add_pose(data[1:])

    def _load_depth(self, frame_id):
        file_index = '0'*(6-len(str(frame_id)))+str(frame_id)
        depth_file = f'{self.dataset_folder}/depth_left/{file_index}_left_depth.npy'
        depth = np.load(depth_file)
        return depth

    def _load_color(self, frame_id):
        file_index = '0'*(6-len(str(frame_id)))+str(frame_id)
        color_file = f'{self.dataset_folder}/image_left/{file_index}_left.png'
        color = cv2.imread(color_file)
        return color
    
    def load_poses(self):
        pose_file = f'{self.dataset_folder}/pose_left.txt'
        with open(pose_file, 'r') as file:
            frame_id = 0
            for line in file:
                data = [float(x) for x in line[:-1].split(' ')]
                data = quaternion_to_rotation_matrix(data[3:]).reshape(9).tolist()+data[:3]
                self.frames[frame_id].add_pose(data)
                frame_id += 1

    def load_depth(self):
        for frame_id, frame in self.frames.items():
            for landmark_id in frame.observed_landmark_id:
                landmark = self.landmarks[landmark_id]
                depth = self._load_depth(frame_id)
                pixel_depth = depth[tuple(reversed([int(x) for x in landmark.measurements[frame_id]]))]
                landmark.add_depth(frame_id, float(pixel_depth))

    def show_matches(self, frame_id, matches):
        curr_color = self._load_color(frame_id)
        next_color = self._load_color(frame_id+1)
        out = np.concatenate([curr_color, next_color], axis=1)
        for match in matches:
            cv2.line(out, match[0], match[1], [255, 0, 0], 1)
        cv2.imshow('matches', out)
        cv2.waitKey(1000)

    def project_landmarks(self):
        fx, fy, cx, cy = self.camera
        for frame_id, curr_frame in self.frames.items():
            print(f'Projecting landmarks in frame {frame_id}')
            # skip the last frame
            if not (frame_id+1) in self.frames:
                continue
            next_frame = self.frames[frame_id+1]
            # iterate over all the landmarks
            matches_gt = []
            matches_slam = []
            for landmark_id in curr_frame.observed_landmark_id:
                landmark = self.landmarks[landmark_id]
                if not frame_id+1 in landmark.measurements:
                    continue
                # point1： frame[i] relative xyz position
                point1 = np.array([*landmark.measurements[frame_id], landmark.depths[frame_id]]).reshape((3,1))
                point1[0] = (point1[0] - cx) * point1[2] / fx
                point1[1] = (point1[1] - cy) * point1[2] / fy
                # point2： frame[i] absolute xyz position
                point2 = self.cameraPose.rotation @ point1 + self.cameraPose.translation
                point2 = curr_frame.rotation @ point1 + curr_frame.translation
                # point3: frame[i+1] absolute xyz position
                point3 = inv(next_frame.rotation) @ (point2 - next_frame.translation)
                # point4: frame[i+1] relative xyz position
                point4 = inv(self.cameraPose.rotation) @ (point3 - self.cameraPose.translation)
                # point5: frame[i+1] uv position
                point5 = [point4[0]/point4[2]*fx+cx, point4[1]/point4[2]*fy+cy]
                landmark.add_gt(frame_id+1, *point5)
                # match
                match = [[],[]]
                match[0] = [int(x) for x in landmark.measurements[frame_id]]
                match[1] = [int(x) for x in [point5[0]+640, point5[1]]]
                matches_gt.append(match)
                match = [[],[]]
                match[0] = [int(x) for x in landmark.measurements[frame_id]]
                next_measurement = landmark.measurements[frame_id+1]
                match[1] = [int(x) for x in [next_measurement[0]+640, next_measurement[1]]]
                matches_slam.append(match)
            # self.show_matches(frame_id, matches_gt)
            self.show_matches(frame_id, matches_slam)
            print(f'number of matches: {len(matches_slam)}')

    def calc_error(self):
        error = 0
        count = 0
        for landmark in self.landmarks.values():
            for frame_id, measurement in landmark.measurements.items():
                if not frame_id in landmark.gt:
                    continue
                gt = landmark.gt[frame_id]
                error += norm(np.array(measurement)-np.array(gt))
                count += 1
        print(f'error: {error/count}')
    
    def evaluate(self, frontend_output):
        self.load_frontend_output(frontend_output)
        self.load_poses()
        self.load_depth()
        self.project_landmarks()
        self.calc_error()

if __name__ == '__main__':
    dataset_type = 'soulcity'
    dataset_folder = os.path.expanduser('~/Projects/curly_slam/data/soulcity')
    fe = FrontendEvaluate(dataset_type, dataset_folder)
    fe.evaluate(os.path.expanduser('~/Projects/curly_slam/data/curly_frontend/test.txt'))
    print("EOF")