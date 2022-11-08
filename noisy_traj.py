'''
Adding Gaussian noise to the trajectory.
'''
import os
from scipy.spatial.transform import Rotation as R
import numpy as np

class NoisyTraj():
    def __init__(self, input_file):
        self.input_file = input_file
        self.poses = {}
        self.load_poses(input_file)

    def load_poses(self, input_file):
        with open(input_file, 'r') as file:
            frame_id = 0
            for line in file:
                data = [float(x) for x in line[:-1].split(' ')]
                rot = R.from_quat(data[3:])
                pos = np.array(data[:3])
                self.poses[frame_id] = (rot, pos)
                frame_id += 1
    
    def add_noise(self, pos_sigma, rot_sigma, output_file):
        curr_rot = self.poses[0][0]
        curr_pos = self.poses[0][1]
        output = open(output_file, 'w')
        # write first pose
        line = np.concatenate([curr_pos, curr_rot.as_quat()])
        output.write(' '.join([str(x) for x in line])+'\n')
        # iter over poses
        rot_euler_noise = np.zeros(3)
        pos_noise = np.zeros(3)
        for frame_id in range(1, len(self.poses)):
            # add noise
            curr_rot = self.poses[frame_id][0]
            curr_pos = self.poses[frame_id][1]
            rot_euler_noise += np.random.normal(0, rot_sigma, 3)
            pos_noise += np.random.normal(0, pos_sigma, 3)
            curr_rot_noisy = R.from_euler('zyx', curr_rot.as_euler('zyx')+rot_euler_noise)
            curr_pos_noisy = R.from_euler('zyx', rot_euler_noise).as_matrix()@pos_noise+curr_pos
            # save traj
            line = np.concatenate([curr_pos_noisy, curr_rot_noisy.as_quat()])
            output.write(' '.join([str(x) for x in line])+'\n')

if __name__ == '__main__':
    dataset_folder = os.path.expanduser('~/Projects/curly_slam/data/soulcity/')
    nt = NoisyTraj(dataset_folder+'pose_left.txt')
    nt.add_noise(0.1, 0.05, dataset_folder+'pose_left_noisy.txt')
    print("EOF")