'''
Using ground truth pose and depth in tartanair dataset to compare visual SLAM frontend feature matching performance.
'''
import os
import cv2
import numpy as np
from numpy.linalg import inv, norm

def quaternion_to_rotation_matrix(Q):
    # from https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/
    # Q: w x y z
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
        self.uv = {}
        self.xyz = {}
        self.depths = {}
        self.gt_uv = {}
        self.gt_xyz = {}
        self.frames = []
    def add_measurement_uv(self, frame_id, u, v):
        self.frames += [frame_id]
        self.uv[frame_id] = [u, v]
    def add_measurement_xyz(self, frame_id, x, y, z):
        self.xyz[frame_id] = [x, y, z]
    def add_depth(self, frame_id, d):
        self.depths[frame_id] = d
    def add_gt(self, frame_id, uv, xyz):
        self.gt_uv[frame_id] = uv
        self.gt_xyz[frame_id] = xyz

class Frame():
    def __init__(self, frame_id):
        self.frame_id = frame_id
        self.dataset_seq = frame_id
        self.observed_landmark_id = []
    def add_landmark_id(self, landmark_id):
        self.observed_landmark_id.append(landmark_id)
    def set_dataset_seq(self, dataset_seq):
        self.dataset_seq = dataset_seq
    def add_odom_pose(self, data):
        rotation = np.array([float(x) for x in data[0:9]]).reshape((3,3))
        translation = np.array([float(x) for x in data[9:12]]).reshape((3,1))
        self.rotation = rotation
        self.translation = translation
    def add_gt_pose(self, data):
        rotation = np.array([float(x) for x in data[0:9]]).reshape((3,3))
        translation = np.array([float(x) for x in data[9:12]]).reshape((3,1))
        self.gt_rotation = rotation
        self.gt_translation = translation

class FrontendEvaluate():
    def __init__(self, dataset_type, dataset_path):
        self.dataset_type = dataset_type
        self.dataset_path = dataset_path
    
    def _in_range(self, frame_id):
        return 

    def load_frontend_output(self, start, end):
        self.landmarks = {}
        self.frames = {}
        self.format = '0'
        self.camera = []
        with open(self.dataset_path['frontend'], 'r') as file:
            current_frame_id = -1
            for line in file:
                data = line[:-1].split(' ')
                if data[0]=='FEATURE':
                    if not current_frame_id>=start and current_frame_id<end:
                        continue
                    landmark_id = int(data[1])
                    if not landmark_id in self.landmarks:
                        self.landmarks[landmark_id] = Landmark(landmark_id)
                    self.frames[current_frame_id].add_landmark_id(landmark_id)
                    self.landmarks[landmark_id].add_measurement_uv(current_frame_id, float(data[2]), float(data[3]))
                    if self.format == '1':
                        self.landmarks[landmark_id].add_depth(current_frame_id, float(data[4]))
                        self.landmarks[landmark_id].add_measurement_xyz(current_frame_id, float(data[5]), float(data[6]), float(data[7]))
                elif data[0]=='DATASET_SEQ':
                    if not current_frame_id>=start and current_frame_id<end:
                        continue
                    self.frames[current_frame_id].set_dataset_seq(int(data[1]));
                elif data[0]=='FRAME':
                    current_frame_id = int(data[1])
                    # only read frames in range of [start, end)
                    if current_frame_id<start:
                        continue
                    elif current_frame_id>=end:
                        break
                    self.frames[current_frame_id]=Frame(current_frame_id)
                elif data[0]=='FORMAT':
                    self.format = data[1]
                    print('Format version: ', self.format)
                elif data[0]=='CAMERA':
                    self.camera = [float(x) for x in data[1:]]
                    print('Loading camera: ', self.camera)
                elif data[0]=='CALIBRATION':
                    self.cameraPose = Frame(-1) # store the pose
                    self.cameraPose.add_odom_pose(data[1:])

    def _load_depth(self, frame_id):
        file_index = '0'*(6-len(str(frame_id)))+str(frame_id)
        depth_file = f'{self.dataset_path["depth"]}/{file_index}_left_depth.npy'
        depth = np.load(depth_file)
        print('loading '+depth_file)
        return depth

    def _load_color(self, frame_id):
        file_index = '0'*(6-len(str(frame_id)))+str(frame_id)
        color_file = f'{self.dataset_path["color"]}/{file_index}_left.png'
        color = cv2.imread(color_file)
        return color
    
    def load_poses(self):
        with open(self.dataset_path["odom_traj"], 'r') as file:
            dataset_seq = -1
            last_frame_id = 0
            lines = file.readlines()
            for frame_id, curr_frame in self.frames.items():
                line = lines[curr_frame.dataset_seq]
                data = [float(x) for x in line[:-1].split(' ')]
                data = quaternion_to_rotation_matrix(data[6:7]+data[3:6]).reshape(9).tolist()+data[:3]
                curr_frame.add_odom_pose(data)
        with open(self.dataset_path["gt_traj"], 'r') as file:
            dataset_seq = -1
            last_frame_id = 0
            lines = file.readlines()
            for frame_id, curr_frame in self.frames.items():
                line = lines[curr_frame.dataset_seq]
                data = [float(x) for x in line[:-1].split(' ')]
                data = quaternion_to_rotation_matrix(data[6:7]+data[3:6]).reshape(9).tolist()+data[:3]
                curr_frame.add_gt_pose(data)

    def load_depth(self):
        for frame_id, frame in self.frames.items():
            depth = self._load_depth(self.frames[frame_id].dataset_seq)
            for landmark_id in frame.observed_landmark_id:
                landmark = self.landmarks[landmark_id]
                pixel_depth = depth[tuple(reversed([int(x) for x in landmark.uv[frame_id]]))]
                landmark.add_depth(frame_id, float(pixel_depth))

    def show_matches(self, window_name, frame_ids, matches, timeout=1000):
        colors = [self._load_color(self.frames[id].dataset_seq) for id in frame_ids]
        out = np.concatenate(colors, axis=1)
        for match in matches:
            points = [[int(x) for x in p] for p in match]
            for i,point in enumerate(points):
                point[0] += colors[i].shape[1]*i
                cv2.circle(out, point, 4, [0, 255, 0])
            for i in range(len(points)-1):
                cv2.line(out, points[i], points[i+1], [255, 0, 0], 1)
        cv2.imshow(window_name, out)
        cv2.waitKey(timeout)

    def project_landmarks(self, viz_matches):
        fx, fy, cx, cy = self.camera
        for frame_id, curr_frame in self.frames.items():
            # print(f'Projecting landmarks in frame {frame_id}')
            # skip the last frame
            if not (frame_id+1) in self.frames:
                continue
            next_frame = self.frames[frame_id+1]
            # iterate over all the landmarks
            matches_gt = []
            matches_slam = []
            for landmark_id in curr_frame.observed_landmark_id:
                landmark = self.landmarks[landmark_id]
                if not frame_id+1 in landmark.uv:
                    continue
                # point0： frame[i] uvd position
                point0 = np.array([*landmark.uv[frame_id], landmark.depths[frame_id]]).reshape((3,1))
                # point1： frame[i] xyz position in cam frame
                point1 = np.array([
                            (point0[0] - cx) * point0[2] / fx,
                            (point0[1] - cy) * point0[2] / fy,
                            point0[2]
                        ])
                # point2： frame[i] xyz position in odom frame
                point2 = self.cameraPose.rotation @ point1 + self.cameraPose.translation
                # point3： frame[i] xyz position in world frame
                point3 = curr_frame.gt_rotation @ point2 + curr_frame.gt_translation
                # point4: frame[i+1] xyz position in odom frame
                point4 = inv(next_frame.gt_rotation) @ (point3 - next_frame.gt_translation)
                # point5: frame[i+1] xyz position in cam frame
                point5 = inv(self.cameraPose.rotation) @ (point4 - self.cameraPose.translation)
                # point6: frame[i+1] uv position
                point6 = [float(point5[0]/point5[2]*fx+cx), float(point5[1]/point5[2]*fy+cy)]
                landmark.add_gt(frame_id+1, point6, point3)
                # match
                matches_gt.append([landmark.uv[frame_id], point6])
                matches_slam.append([landmark.uv[frame_id], landmark.uv[frame_id+1]])
                if len(landmark.gt_uv)>5:
                    tmp = 123
            if len(matches_gt)>0 and viz_matches:
                self.show_matches('slam', [frame_id, frame_id+1], matches_slam)
                self.show_matches('gt', [frame_id, frame_id+1], matches_gt, 2000)
                print(f'number of matches: {len(matches_slam)}')

    def calc_error(self):
        error = 0
        count = 0
        for landmark in self.landmarks.values():
            for frame_id, measurement in landmark.uv.items():
                if not frame_id in landmark.gt_uv:
                    continue
                gt = landmark.gt_uv[frame_id]
                error += norm(np.array(measurement)-np.array(gt))
                count += 1
            if len(landmark.xyz)>1:
                xxxx=1
            # if landmark.landmark_id==1647:
            #     xxxx=1
        print(f'error: {error/count}')
    
    def load_dataset(self, start=0, end=10000):
        self.load_frontend_output(start, end)
        self.load_poses()
        if self.format == '0':
            self.load_depth()
    
    def evaluate(self, viz_matches=False):
        self.project_landmarks(viz_matches)
        self.calc_error()

if __name__ == '__main__':
    dataset_type = 'tartanair'
    dataset_folder = os.path.expanduser('~/Projects/curly_slam/data/tartanair/scenes/seasidetown/Easy/P000')
    frontend_file = os.path.expanduser('~/Projects/curly_slam/data/curly_frontend/curly_tartanair_seasidetown_000.txt')
    dataset_path = {
        'depth': dataset_folder+'/depth_image',
        'color': dataset_folder+'/image_left',
        'frontend': frontend_file,
        'gt_traj': dataset_folder+'/pose_left.txt',
        'odom_traj': dataset_folder+'/pose_left_noisy.txt'
    }
    
    frontend = FrontendEvaluate(dataset_type, dataset_path)
    frontend.load_dataset(start=0, end=500)
    frontend.evaluate(viz_matches=False)
    print("EOF")