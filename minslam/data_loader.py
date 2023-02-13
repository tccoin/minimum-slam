import numpy as np
import cv2
from evo.tools import file_interface
from evo.core.trajectory import PoseTrajectory3D, PosePath3D
import os
from scipy.spatial.transform import Rotation as R


class DataLoader():
    def __init__(self, dataset_folder):
        self.dataset_folder = os.path.expanduser(dataset_folder)
        if self.dataset_folder[-1] != '/':
            self.dataset_folder += '/'
        self.depth_folder = 'depth_left'
        self.stereo_folders = ['image_left', 'image_right']
        self.gt_filename = 'pose_left.txt'
        self.odom_filename = 'pose_left.txt'
        self.current_index = 0
        self.index_interval = 1
        self.end_index = -1
        self.camera = [0, 0, 0, 0]  # fx, fy, cx, cy
        self.pose_odom_cam = np.eye(4)  # p_odom = T_cam_odom * p_cam
        self.image_size = (0, 0)  # width, height

    def read_current_rgbd(self):
        raise NotImplementedError()

    def read_current_stereo(self):
        raise NotImplementedError()

    def read_ground_truth(self):
        raise NotImplementedError()

    def read_odometry(self):
        raise NotImplementedError()

    def get_total_number(self):
        '''
        count number of frames according to number of files in color folder
        '''
        dir_path = self.dataset_folder + self.stereo_folders[0]
        return len([entry for entry in os.listdir(dir_path)
                    if os.path.isfile(os.path.join(dir_path, entry))])

    def _load_traj(self, traj_type, traj_filename, ignore_timestamps=False, add_timestamps=False):
        '''
        load trajectory from file
        @param traj_type: ['kitti', 'tum', 'euroc']
        @param ignore_timestamps: if True, ignore the timestamps in the first column
        @param add_timestamps: if True, add timestamps to the first column
        @return: evo.core.trajectory.PoseTrajectory3D
        '''
        function_dict = {
            'kitti': file_interface.read_kitti_poses_file,
            'tum': file_interface.read_tum_trajectory_file,
            'euroc': file_interface.read_euroc_csv_trajectory
        }
        file_path = self.dataset_folder + traj_filename
        if ignore_timestamps or add_timestamps:
            if not os.path.exists('tmp'):
                os.makedirs('tmp')
            tmp_file_path = 'tmp/tmp_' + traj_filename
            with open(file_path, 'r') as raw_file:
                lines = raw_file.readlines()
            with open(tmp_file_path, 'w') as output_file:
                for i, line in enumerate(lines):
                    if ignore_timestamps:
                        new_line = ' '.join(line.split(' ')[1:])
                    elif add_timestamps:
                        new_line = str(i)+' '+line
                    output_file.write(new_line)
            file_path = tmp_file_path
        traj = function_dict[traj_type](file_path)
        if ignore_timestamps or add_timestamps:
            os.remove(tmp_file_path)
        return traj

    def _zeros(self, str_length, num):
        '''
        @param str_length: length of the string
        @param num: number of zeros
        @return: a string with leading zeros
        '''
        return '0' * (str_length - len(str(num))) + str(num)

    def get_current_index(self):
        return self.current_index

    def set_current_index(self, index):
        self.current_index = index

    def load_next_frame(self):
        '''
        @return: True if there are still frames to load
        '''
        self.current_index += self.index_interval
        if self.end_index > 0:
            return self.current_index <= self.end_index
        else:
            return True

    def add_noise(self, traj, mean_sigma=[5e-3, 5e-3], sigma=[1e-2, 1e-2], seed=None):
        '''
        add noise to the trajectory
        @param traj: evo.core.trajectory.PoseTrajectory3D
        @param mean_sigma: standard deviation of the mean of the noise [translation, rotation]
        @param sigma: standard deviation of the noise [translation, rotation]
        @param seed: random seed
        @return: evo.core.trajectory.PosePath3D
        '''
        if seed is None:
            seed = np.random.randint(0, 100000)
            print(f'Adding noise, seed={seed}')
        np.random.seed(seed)
        new_traj = [traj.poses_se3[0]]
        noise_t = np.zeros([3, 1])
        noise_r = np.eye(3)
        noise_t_bias = np.random.normal(0, mean_sigma[0], (3, 1))
        noise_r_bias = R.from_euler('xyz', np.random.normal(
            0, mean_sigma[1], 3)).as_matrix()
        for i in range(1, len(traj.poses_se3)):
            noise_t_delta = noise_r @ (np.random.normal(
                0, sigma[0], (3, 1)) + noise_t_bias)
            noise_r_delta = R.from_euler('xyz', np.random.normal(
                0, sigma[1], 3)).as_matrix() @ noise_r_bias
            noise_r = noise_r_delta @ noise_r
            noise_t = noise_r_delta @ noise_t_delta + noise_t
            new_pose = np.eye(4)
            new_pose[:3, 3:4] = noise_t + traj.poses_se3[i][:3, 3:4]
            new_pose[:3, :3] = noise_r @ traj.poses_se3[i][:3, :3]
            new_traj += [new_pose]
        return PosePath3D(poses_se3=new_traj)


class TartanAirLoader(DataLoader):
    def __init__(self, dataset_folder, depth_folder='depth_left/'):
        super().__init__(dataset_folder)
        self.depth_folder = depth_folder
        self.stereo_folders = ['image_left/', 'image_right/']
        self.gt_filename = 'pose_left.txt'
        self.odom_filename = 'pose_left.txt'
        self.camera = [320, 320, 320, 240]  # fx, fy, cx, cy
        self.pose_odom_cam = np.array([
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])  # p_odom = T_odom_cam * p_cam
        self.image_size = (640, 480)  # width, height

    def read_current_rgbd(self):
        index_str = super()._zeros(6, self.current_index)
        left_color = cv2.imread(
            f'{self.dataset_folder}{self.stereo_folders[0]}{index_str}_left.png')
        left_depth = np.load(
            f'{self.dataset_folder}{self.depth_folder}{index_str}_left_depth.npy')
        return (left_color, left_depth)

    def read_current_stereo(self):
        index_str = self._zeros(6, self.current_index)
        left_color = cv2.imread(
            f'{self.dataset_folder}{self.stereo_folders[0]}{index_str}_left.png')
        right_color = cv2.imread(
            f'{self.dataset_folder}{self.stereo_folders[1]}{index_str}_right.png')
        return (left_color, right_color)

    def read_ground_truth(self):
        self.gt = self._load_traj('tum', 'pose_left.txt', add_timestamps=True)
        return self.gt

    def read_odometry(self):
        self.odom = self._load_traj(
            'tum', 'pose_left.txt', add_timestamps=True)
        return self.odom


if __name__ == '__main__':
    dataset_folder = './test/traj_examples'
    data_loader = DataLoader(dataset_folder)
    traj = data_loader._DataLoader__load_traj(
        'tum', 'tum_no_timestamp.txt', add_timestamps=True)
    print(traj.num_poses)
    traj = data_loader._DataLoader__load_traj(
        'kitti', 'kitti_with_timestamp.txt', ignore_timestamps=True)
    print(traj)
