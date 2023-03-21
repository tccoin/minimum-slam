'''
Using ground truth pose and depth in tartanair dataset to compare visual SLAM frontend feature matching performance.
'''
import os
import cv2
import numpy as np
from numpy.linalg import inv, norm
from copy import deepcopy


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
        self.match_error = {}
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

    def add_gt_xyz(self, frame_id, xyz):
        self.gt_xyz[frame_id] = xyz

    def add_gt_uv(self, frame_id, uv):
        self.gt_uv[frame_id] = uv

    def add_match_error(self, frame_id, err):
        self.match_error[frame_id] = err
    



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
        # x y z qx qx qz qw
        data = [float(x) for x in data]
        self.rotation = quaternion_to_rotation_matrix([data[6]]+data[3:6])
        self.translation = np.array([float(x) for x in data[:3]]).reshape([3, 1])

    def add_gt_pose(self, data):
        data = [float(x) for x in data]
        self.gt_rotation = quaternion_to_rotation_matrix([data[6]]+data[3:6])
        self.gt_translation = np.array([float(x) for x in data[:3]]).reshape([3, 1])
        x=1

    def align_gt(self, gt_rot, gt_pos, rot, pos):
        self.gt_translation = rot@(np.linalg.inv(gt_rot)
                                   @ (self.gt_translation-gt_pos))+pos
        self.gt_rotation = rot@(np.linalg.inv(gt_rot)@self.gt_rotation)


class FrontendEvaluate():
    def __init__(self, dataset_type, dataset_path):
        self.dataset_type = dataset_type
        self.dataset_path = dataset_path
        self.__gt_xyz_calculated = False

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
                if data[0] == 'FEATURE':
                    # skip the frame
                    if not current_frame_id >= start and current_frame_id < end:
                        continue
                    # skip measurement with inf depth
                    if float(data[4]) > 65:
                        continue
                    # save the measurement
                    landmark_id = int(data[1])
                    if not landmark_id in self.landmarks:
                        self.landmarks[landmark_id] = Landmark(landmark_id)
                    self.frames[current_frame_id].add_landmark_id(landmark_id)
                    self.landmarks[landmark_id].add_measurement_uv(
                        current_frame_id, float(data[2]), float(data[3]))
                    self.landmarks[landmark_id].add_depth(
                        current_frame_id, float(data[4]))
                    self.landmarks[landmark_id].add_measurement_xyz(
                        current_frame_id, float(data[5]), float(data[6]), float(data[7]))
                elif data[0] == 'DATASET_SEQ':
                    if not current_frame_id >= start and current_frame_id < end:
                        continue
                    self.frames[current_frame_id].set_dataset_seq(int(data[1]))
                elif data[0] == 'FRAME':
                    current_frame_id = int(data[1])
                    # only read frames in range of [start, end)
                    if current_frame_id < start:
                        continue
                    elif current_frame_id >= end:
                        break
                    self.frames[current_frame_id] = Frame(current_frame_id)
                elif data[0] == 'FORMAT':
                    self.format = data[1]
                    print('Format version: ', self.format)
                    if self.format != '2':
                        print('Warning: this format is deprecated, please use the latest version')
                        return
                elif data[0] == 'CAMERA_INTRINSIC':
                    self.camera = [float(x) for x in data[1:]]
                    print('Loading camera: ', self.camera)
                elif data[0] == 'CAMERA_POSE':
                    self.cameraPose = Frame(-1)  # store the pose
                    self.cameraPose.add_odom_pose(data[1:])
                    # print('camera pose: \n', self.cameraPose.rotation,'\n', self.cameraPose.translation)

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

    def load_poses(self, align_start_point):
        self.load_odom_traj()
        self.load_gt_traj()
        if align_start_point:
            self.align_start_point()

    def load_odom_traj(self):
        with open(self.dataset_path["odom_traj"], 'r') as file:
            dataset_seq = -1
            lines = file.readlines()
            for frame_id, curr_frame in self.frames.items():
                dataset_seq = min(len(lines)-1, curr_frame.dataset_seq)
                line = lines[dataset_seq]
                data = line[:-1].split(' ')
                curr_frame.add_odom_pose(data)

    def load_gt_traj(self):
        with open(self.dataset_path["gt_traj"], 'r') as file:
            dataset_seq = -1
            lines = file.readlines()
            for frame_id, curr_frame in self.frames.items():
                dataset_seq = min(len(lines), curr_frame.dataset_seq)
                line = lines[dataset_seq]
                data = line[:-1].split(' ')
                curr_frame.add_gt_pose(data)

    def align_start_point(self):
        first_frame = next(iter(self.frames.values()))
        gt_rot = first_frame.gt_rotation
        gt_pos = first_frame.gt_translation
        rot = first_frame.rotation
        pos = first_frame.translation
        for frame_id, curr_frame in self.frames.items():
            curr_frame.align_gt(gt_rot, gt_pos, rot, pos)

    def load_depth(self):
        for frame_id, frame in self.frames.items():
            depth = self._load_depth(self.frames[frame_id].dataset_seq)
            for landmark_id in frame.observed_landmark_id:
                landmark = self.landmarks[landmark_id]
                pixel_depth = depth[tuple(
                    reversed([int(x) for x in landmark.uv[frame_id]]))]
                landmark.add_depth(frame_id, float(pixel_depth))

    def show_matches(self, window_name, frame_ids, matches, match_color=None, timeout=1000, match_info=None, max_len=100):
        frame_ids = frame_ids[-max_len:]
        for i in range(len(matches)):
            matches[i] = matches[i][-max_len:]
        if match_info is None:
            match_info = [['']*len(frame_ids)]*len(matches)
        if match_color is None:
            match_color = [np.random.randint(0,256,3) for i in range(len(matches))]
        colors = [self._load_color(self.frames[id].dataset_seq)
                  for id in frame_ids]
        out = np.concatenate(colors, axis=1)
        print(f'Visualizing matches, frame: [{frame_ids[0]}, {frame_ids[-1]}]')
        for match, info, color in zip(matches, match_info, match_color):
            points = [[int(x) for x in p] for p in match]
            color = [int(x) for x in color]
            for i, point in enumerate(points):
                point[0] += colors[i].shape[1]*i
                cv2.circle(out, point, 4, color)
            for i in range(len(points)-1):
                cv2.line(out, points[i], points[i+1], color, 1)
            for point, text in zip(points, info):
                text_pos = [point[0], point[1]-5]
                cv2.putText(out, str(text), point,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, out)
        cv2.waitKey(timeout)

    def project_landmarks(self):
        # self.calc_ground_truth_xyz()
        self.calc_ground_truth_match()

    def calc_ground_truth_xyz(self):
        fx, fy, cx, cy = self.camera
        count = 0
        for frame_id, curr_frame in self.frames.items():
            for landmark_id in curr_frame.observed_landmark_id:
                landmark = self.landmarks[landmark_id]
                # point0： frame[i] uvd position
                point0 = np.array(
                    [*landmark.uv[frame_id], landmark.depths[frame_id]]).reshape((3, 1))
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
                landmark.add_gt_xyz(frame_id, point3)
                count += 1
        self.__gt_xyz_calculated = True
        print(f'Ground truth xyz calculated, count={count}')

    def calc_ground_truth_match(self):
        if not self.__gt_xyz_calculated:
            self.calc_ground_truth_xyz()
        fx, fy, cx, cy = self.camera
        count = 0
        for landmark in self.landmarks.values():
            frames = sorted(landmark.frames)
            for i in range(len(frames)):
                frame_id = frames[i]
                if i == 0:
                    # if it's the first frame, gt uv is the original uv
                    landmark.add_gt_uv(frame_id, landmark.uv[frame_id])
                    count += 1
                if i==len(frames)-1:
                    break
                else:
                    # otherwise, project the uv to the next frame
                    next_frame_id = frames[i+1]
                    curr_frame = self.frames[frame_id]
                    next_frame = self.frames[next_frame_id]
                    # read point3： frame[i] xyz position in world frame
                    point3 = landmark.gt_xyz[frames[i]]
                    # point4: frame[i+1] xyz position in odom frame
                    point4 = inv(next_frame.gt_rotation) @ (point3 -
                                                            next_frame.gt_translation)
                    # point5: frame[i+1] xyz position in cam frame
                    point5 = inv(self.cameraPose.rotation) @ (point4 -
                                                            self.cameraPose.translation)
                    # point6: frame[i+1] uv position
                    point6 = [float(point5[0]/point5[2]*fx+cx),
                            float(point5[1]/point5[2]*fy+cy)]
                    landmark.add_gt_uv(next_frame_id, point6)
                    count += 1
        print(f'Ground truth match calculated, count={count}')

    def calc_error(self):
        error = 0
        count = 0
        total = 0
        for landmark in self.landmarks.values():
            for frame_id, measurement in landmark.uv.items():
                if not frame_id in landmark.gt_uv:
                    continue
                gt = landmark.gt_uv[frame_id]
                err = norm(np.array(measurement)-np.array(gt))
                landmark.add_match_error(frame_id, err)
                error += err
                total += 1
                if err > 10:
                    count += 1
        print(
            f'error per frame: {error/len(self.frames)}, error per match: {error/total}, error>10:{count}/{total}, frame={len(self.frames)}')

    def load_dataset(self, start=0, end=10000, align_start_point=True):
        self.load_frontend_output(start, end)
        self.load_poses(align_start_point)
        if self.format == '0':
            self.load_depth()

    def check_size(self):
        print(f'Number of frames: {len(self.frames)}')
        print(f'Number of landmarks: {len(self.landmarks)}')

        count = 0
        for landmark in self.landmarks.values():
            for frame_id in landmark.uv.keys():
                count += 1
        print(f'Number of measurements: {count}')

        all_measurements_has_gt = True
        for landmark in self.landmarks.values():
            for frame_id in landmark.uv.keys():
                if not frame_id in landmark.gt_xyz:
                    all_measurements_has_gt = False
                    break
                if not frame_id in landmark.gt_uv:
                    all_measurements_has_gt = False
                    break
        print(f'All measurements has ground truth: {all_measurements_has_gt}')


    def evaluate(self, viz_matches=False):
        self.project_landmarks()
        self.calc_error()
        self.check_size()

    def visualize_outliers(self, error_threshold=10):
        for landmark in self.landmarks.values():
            frames = []
            matches = [[],[]]
            match_info = [[],[]]
            outlier_frame_index = -1
            for i in range(len(landmark.frames)):
                frame_id = landmark.frames[i]
                if landmark.match_error[frame_id] > error_threshold:
                    outlier_frame_index = i
            if outlier_frame_index>-1:
                max_len = 5
                frames = landmark.frames[max(0, outlier_frame_index+1-max_len):outlier_frame_index+1]
                for frame_id in frames:
                    if len(frames)==1:
                        print(f'frame_id={frame_id}, landmark_id={landmark.landmark_id}, error={landmark.match_error[frame_id]}')
                    matches[0].append(landmark.uv[frame_id])
                    match_info[0].append(landmark.depths[frame_id])
                    matches[1].append(landmark.gt_uv[frame_id])
                    match_info[1].append('')
                    # match_info[1].append(landmark.depths[frame_id])
                self.show_matches('outlier', frames, matches, match_info=match_info, match_color=[[255,100,100], [100,100,255]], timeout=-1)


if __name__ == '__main__':
    dataset_type = 'tartanair'
    dataset_folder = os.path.expanduser(
        '~/Projects/curly_slam/data/tartanair/scenes/abandonedfactory/Easy/P001')
    frontend_file = os.path.expanduser(
        '~/Projects/curly_slam/data/log/abandonedfactory_easy_p001.txt')
    dataset_path = {
        'depth': dataset_folder+'/depth_left',
        'color': dataset_folder+'/image_left',
        'frontend': frontend_file,
        'gt_traj': dataset_folder+'/pose_left.txt',
        'odom_traj': dataset_folder+'/pose_left.txt'
    }

    frontend = FrontendEvaluate(dataset_type, dataset_path)
    frontend.load_dataset(start=0, end=500)
    frontend.evaluate(viz_matches=False)
    # frontend.visualize_outliers(error_threshold=20)
    print("EOF")