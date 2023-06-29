import os
import numpy as np
import cv2
from evo.tools import file_interface
from spatialmath import *
import plotly.graph_objects as go


class DataLoaderBase():
    def __init__(self, dataset_folder):
        self.dataset_folder = self._fix_path(
            os.path.expanduser(dataset_folder))
        self.depth_folder = 'depth_left'
        self.stereo_folders = ['image_left', 'image_right']
        self.gt_filename = 'pose_left.txt'
        self.odom_filename = 'pose_left.txt'
        self.curr_index = 0
        self.index_interval = 1
        self.end_index = -1
        self.camera = [0, 0, 0, 0]  # fx, fy, cx, cy
        self.pose_odom_cam = np.eye(4)  # p_odom = T_cam_odom * p_cam
        self.image_size = (0, 0)  # width, height

    def read_current_rgbd(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def read_current_stereo(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def read_current_ground_truth(self) -> SE3:
        raise NotImplementedError()

    def read_current_odometry(self) -> SE3:
        raise NotImplementedError()

    def load_ground_truth(self) -> None:
        raise NotImplementedError()

    def load_odometry(self) -> None:
        raise NotImplementedError()

    def set_odometry(self, traj) -> None:
        raise NotImplementedError()

    def get_total_number(self) -> int:
        '''
        count number of frames according to number of files in color folder
        '''
        dir_path = self.dataset_folder + self.stereo_folders[0]
        return len([entry for entry in os.listdir(dir_path)
                    if os.path.isfile(os.path.join(dir_path, entry))])

    def _load_traj(self, traj_type, traj_filename, ignore_timestamps=False, add_timestamps=False) -> SE3:
        '''
        load trajectory from file
        @param traj_type: ['kitti', 'tum', 'euroc']
        @param ignore_timestamps: if True, ignore the timestamps in the first column
        @param add_timestamps: if True, add timestamps to the first column
        @return: SE3
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
        traj_evo = function_dict[traj_type](file_path)
        # convert PoseTrajectory3D to SE3
        traj = SE3(traj_evo.poses_se3)
        # remove tmp file
        if ignore_timestamps or add_timestamps:
            os.remove(tmp_file_path)
        return traj

    def _zeros(self, str_length, num) -> str:
        '''
        @param str_length: length of the string
        @param num: number of zeros
        @return: a string with leading zeros
        '''
        return '0' * (str_length - len(str(num))) + str(num)

    def _fix_path(self, path) -> str:
        '''
        add a / at the end of the path if it is not there
        '''
        return path if path[-1] == '/' else path+'/'

    def get_curr_index(self) -> int:
        return self.curr_index

    def set_curr_index(self, index) -> None:
        self.curr_index = index

    def load_next_frame(self) -> bool:
        '''
        @return: True if there are still frames to load
        '''
        self.curr_index += self.index_interval
        if self.end_index > 0:
            return self.curr_index <= self.end_index
        else:
            return True

    def add_noise(self, traj, mean_sigma=[2e-4, 2e-4], sigma=[1e-3, 1e-3], seed=None, start=0) -> SE3:
        '''
        add noise to the trajectory
        @param traj: SE3
        @param mean_sigma: standard deviation of the mean of the noise [translation, rotation]
        @param sigma: standard deviation of the noise [translation, rotation]
        @param seed: random seed
        @return: SE3
        '''
        if seed is None:
            seed = np.random.randint(0, 100000)
            print(f'Adding noise, seed={seed}')
        np.random.seed(seed)
        new_traj = SE3(traj)
        noise = SE3()
        noise_t_bias = SE3.Trans(np.random.normal(0, mean_sigma[0], 3))
        noise_r_bias = SE3.RPY(*np.random.normal(0, mean_sigma[1], 3))
        for i in range(start+1, len(traj)):
            noise_t_delta = SE3.Trans(
                np.random.normal(0, sigma[0], 3)) * noise_t_bias
            noise_r_delta = SE3.RPY(
                *np.random.normal(0, sigma[1], 3)) * noise_r_bias
            noise = noise_r_delta * noise_t_delta * noise
            new_pose = noise * new_traj[i]
            new_traj[i] = new_pose
        return new_traj


class TartanAirLoader(DataLoaderBase):
    def __init__(self, dataset_folder, depth_folder='depth_left/',
                 stereo_folders_left='image_left/', stereo_folders_right='image_right/'):
        super().__init__(dataset_folder)
        self.depth_folder = self._fix_path(depth_folder)
        self.stereo_folders = [
            self._fix_path(stereo_folders_left),
            self._fix_path(stereo_folders_right)
        ]
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

    def read_current_rgbd(self) -> tuple[np.ndarray, np.ndarray]:
        index_str = super()._zeros(6, self.curr_index)
        left_color = cv2.imread(
            f'{self.dataset_folder}{self.stereo_folders[0]}{index_str}_left.png')
        left_depth = np.load(
            f'{self.dataset_folder}{self.depth_folder}{index_str}_left_depth.npy')
        return (left_color, left_depth)

    def read_current_stereo(self) -> tuple[np.ndarray, np.ndarray]:
        index_str = self._zeros(6, self.curr_index)
        left_color = cv2.imread(
            f'{self.dataset_folder}{self.stereo_folders[0]}{index_str}_left.png')
        right_color = cv2.imread(
            f'{self.dataset_folder}{self.stereo_folders[1]}{index_str}_right.png')
        return (left_color, right_color)

    def read_current_ground_truth(self) -> SE3:
        return self.gt[self.curr_index]

    def read_current_odometry(self) -> SE3:
        return self.odom[self.curr_index]

    def load_ground_truth(self) -> None:
        poses = self._load_traj('tum', 'pose_left.txt', add_timestamps=True)
        T = np.array([[0,1,0,0],
                        [0,0,1,0],
                        [1,0,0,0],
                        [0,0,0,1]], dtype=np.float32) 
        T_inv = np.linalg.inv(T)
        gt = []
        for t in poses.data:
            gt.append(SE3(T.dot(t).dot(T_inv)))
        self.gt = SE3(gt)

    def set_ground_truth(self, traj) -> None:
        self.gt = traj

    def load_odometry(self, traj=None) -> None:
        self.odom = self._load_traj(
            'tum', 'pose_left.txt', add_timestamps=True)

    def set_odometry(self, traj) -> None:
        self.odom = traj


def load_dataset(params):
    if params['dataset']['type'] == 'tartanair':
        return TartanAirLoader(
            params['dataset']['folder'],
            depth_folder=params['dataset']['depth']
        )
    else:
        raise NotImplementedError

def plot_trajectory(trajectory, legend_name='trajectory', fig=None):
    if fig is None:
        fig = go.Figure()
    trajectory = np.array(trajectory)
    X, Y, Z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
    fig.add_trace(go.Scatter3d(x=X, y=Y, z=Z, mode='lines', name=legend_name))
    fig.update_layout(
        margin=dict(r=20, l=10, b=10, t=10),
        legend=dict(y=0.5, traceorder='reversed', font_size=16),
        autosize=True,
    )
    return fig