'''
playground for testing GTSAM
'''
import os
import gtsam
import numpy as np
from gtsam.symbol_shorthand import L, X
import matplotlib.pyplot as plt
from frontend_evaluate import FrontendEvaluate

class GTSAM_Backend():
    def __init__(self, frontend):
        self.frontend = frontend
    
    def optimize(self, use_smart_factor=True):
        # graph
        graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()

        # factors
        smart_factors = {}

        # camera and noise
        fx, fy, cx, cy = self.frontend.camera
        K = gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)
        uv_measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 30)
        landmark_prior_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        pose_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6)*1)

        # create projection factors
        for frame_id, curr_frame in self.frontend.frames.items():
            curr_frame = self.frontend.frames[frame_id]
            for landmark_id in curr_frame.observed_landmark_id:
                landmark = self.frontend.landmarks[landmark_id]
                if use_smart_factor:
                    if landmark.frames[0]==frame_id:
                        factor = gtsam.SmartProjectionPose3Factor(uv_measurement_noise, K)
                        smart_factors[landmark_id] = factor
                        graph.add(factor)
                    else:
                        factor = smart_factors[landmark_id]
                    factor.add(np.array(landmark.uv[frame_id]), X(frame_id))
                else:
                    if landmark.frames[0]==frame_id:
                        initial_estimate.insert(L(landmark_id), gtsam.Point3(*landmark.xyz[frame_id]))
                        graph.push_back(gtsam.PriorFactorPoint3(
                            L(landmark_id), gtsam.Point3(*landmark.xyz[frame_id]), landmark_prior_noise
                        ))
                    graph.push_back(gtsam.GenericProjectionFactorCal3_S2(
                        np.array(landmark.uv[frame_id]), uv_measurement_noise, X(frame_id),
                        L(landmark_id), K
                    ))
            pose = gtsam.Pose3(
                gtsam.Rot3(curr_frame.rotation),
                gtsam.Point3(curr_frame.translation.flatten())
            )
            initial_estimate.insert(X(frame_id), pose)
            graph.push_back(gtsam.PriorFactorPose3(
                X(frame_id), pose, pose_prior_noise
            ))
                

        # batch optimize
        # optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
        optimizer = gtsam.GaussNewtonOptimizer(graph, initial_estimate)
        self.current_estimate = optimizer.optimize()
        # try:
        #     self.current_estimate = optimizer.optimize()
        # except:
        #     print('Optimization failed')

        # plot
        self.plot()

    def plot(self, plot_input_traj=True, plot_estimated_traj=True, plot_gt_traj=False):
        # setup canvas
        # fignum = 0
        # fig = plt.figure(fignum)
        # ax = plt.axes(projection='3d')
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        # plt.cla()
        # axes.set_xlim3d(-1, 1)
        # axes.set_ylim3d(-1, 1)
        # axes.set_zlim3d(-1, 1)
        first_id = next(iter(self.frontend.frames))

        # plot input_traj
        if plot_input_traj:
            traj = np.zeros([3, len(self.frontend.frames)])
            for frame_id, curr_frame in self.frontend.frames.items():
                traj[:, frame_id-first_id] = curr_frame.translation.flatten()
            ax.plot3D(traj[0,:], traj[1,:], traj[2,:], 'blue', label='input')

        # plot estimated_traj
        if plot_estimated_traj:
            traj = np.zeros([3, len(self.frontend.frames)])
            for frame_id, curr_frame in self.frontend.frames.items():
                traj[:, frame_id-first_id] = self.current_estimate.atPose3(X(frame_id)).translation()
            ax.plot3D(traj[0,:], traj[1,:], traj[2,:], 'red', label='estimated')

        # show the plot
        ax.legend()
        plt.show()
        # plt.pause(100)

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
    frontend.load_dataset(start=0, end=30)

    backend = GTSAM_Backend(frontend)
    backend.optimize()
    print("EOF")