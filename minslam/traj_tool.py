'''
Adding Gaussian noise to the trajectory.
'''
import os
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt

class TrajTool():
    def __init__(self, input_file, part=None):
        self.input_file = input_file
        self.poses = {}
        self.part = part
        self.load_poses(input_file)

        # plot
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        self.ax = ax
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    def load_poses(self, input_file):
        with open(input_file, 'r') as file:
            frame_id = 0
            for line in file:
                if self.part is None or (frame_id>=self.part[0] and frame_id<self.part[1]):
                    data = [float(x) for x in line[:-1].split(' ')]
                    rot = R.from_quat(data[3:])
                    pos = np.array(data[:3])
                    self.poses[frame_id] = (rot, pos)
                frame_id += 1
    
    def add_noise(self, mean_pos_sigma, mean_rot_sigma, pos_sigma, rot_sigma, output_file, plot=True):
        curr_rot = self.poses[0][0]
        curr_pos = self.poses[0][1]
        output = open(output_file, 'w')
        # write first pose
        line = np.concatenate([curr_pos, curr_rot.as_quat()])
        output.write(' '.join([str(x) for x in line])+'\n')
        # iter over poses
        rot_euler_noise = np.zeros(3)
        pos_noise = np.zeros(3)
        mean_euler = np.random.normal(0, mean_rot_sigma, 3)
        mean_pos = np.random.normal(0, mean_pos_sigma, 3)
        input_traj = []
        noisy_traj = []
        for frame_id in range(1, len(self.poses)):
            # add noise
            curr_rot = self.poses[frame_id][0]
            curr_pos = self.poses[frame_id][1]
            rot_euler_noise += np.random.normal(mean_euler, rot_sigma, 3)
            pos_noise += np.random.normal(mean_pos, pos_sigma, 3)
            curr_rot_noisy = R.from_euler('zyx', curr_rot.as_euler('zyx')+rot_euler_noise)
            curr_pos_noisy = R.from_euler('zyx', rot_euler_noise).as_matrix()@pos_noise+curr_pos
            input_traj += [curr_pos]
            noisy_traj += [curr_pos_noisy]
            # save traj
            line = np.concatenate([curr_pos_noisy, curr_rot_noisy.as_quat()])
            output.write(' '.join([str(x) for x in line])+'\n')
        self.plot(input_traj, 'blue', 'input')
        self.plot(noisy_traj, 'red', 'noisy', True)
    
    def transform(self, rot, pos, output_file, ref_file, plot=True):
        output = open(output_file, 'w')
        # iter over poses
        input_traj = []
        transformed_traj = []
        for frame_id, curr_pose in self.poses.items():
            curr_rot, curr_pos = curr_pose
            transformed_rot = rot*curr_rot
            # transformed_pos = rot.as_matrix()@curr_pos+pos
            # transformed_rot = curr_rot
            transformed_pos = curr_pos
            input_traj += [curr_pos]
            transformed_traj += [transformed_pos]
            if frame_id%20==0:
                self.plot_axis(transformed_rot, transformed_pos)
                tmp1 = transformed_rot.as_matrix()
                xxx=213
            # save traj
            line = np.concatenate([transformed_pos, transformed_rot.as_quat()])
            output.write(' '.join([str(x) for x in line])+'\n')
        self.plot(transformed_traj, 'red', 'transformed')
        self.load_poses(ref_file)
        for frame_id, curr_pose in self.poses.items():
            if frame_id%20==0:
                self.plot_axis(self.poses[frame_id][0], self.poses[frame_id][1])
        self.plot([pos for _,pos in self.poses.values()], 'blue', 'ref', True)
    
    def add_fake_timestamp(self, output_file):
        output = open(output_file, 'w')
        # iter over poses
        for frame_id, curr_pose in self.poses.items():
            curr_rot, curr_pos = curr_pose
            line = np.concatenate([[frame_id], curr_pos, curr_rot.as_quat()])
            output.write(' '.join([str(x) for x in line])+'\n')

    def plot_axis(self, rot, pos):
        point1 = (rot.as_matrix()@np.array([1,0,0])).flatten() + pos
        print(point1, pos, rot.as_matrix())
        self.ax.plot3D(xs=[point1[0], pos[0]], ys=[point1[1], pos[1]], zs=[point1[2], pos[2]])

    def plot(self, pos_list, color, label, show=False):
        ax = self.ax
        traj = np.zeros([3, len(pos_list)])
        for i, pos in enumerate(pos_list):
            traj[:, i] = pos
        ax.plot3D(traj[0,:], traj[1,:], traj[2,:], color, label=label)
        ax.scatter(traj[0,0:1], traj[1,0:1], traj[2,0:1], marker='o')
        if show:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.legend()
            plt.show()

if __name__ == '__main__':
    dataset_folder = os.path.expanduser('~/Projects/curly_slam/data/tartanair/scenes/oldtown/Easy/P000')
    tool = TrajTool(dataset_folder+'/pose_left.txt')
    # tool = TrajTool(dataset_folder+'/cvo_timestamp.tum')
    tool.add_noise(0.001, 0.002, 0.005, 0.002, dataset_folder+'/pose_left_noisy.txt')
    
    # tool.transform(R.from_euler('zyx', [np.pi/2, np.pi/2, 0]), np.zeros(3), dataset_folder+'/cvo_pose_transformed.txt', dataset_folder+'/pose_left.txt')
    # tool.add_fake_timestamp(dataset_folder+'/gt_timestamp.txt')

    # tool.transform(R.from_euler('zyx', [-np.pi/2, 0, -np.pi/2]), np.zeros(3), dataset_folder+'/cvo_pose_transformed.txt', dataset_folder+'/pose_left.txt')

    # tool = TrajTool(dataset_folder+'/cvo_pose_transformed.txt')
    # tool.add_fake_timestamp(dataset_folder+'/cvo_timestamp.txt')
    print("EOF")