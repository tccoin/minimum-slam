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
    
    def optimize(self):
        graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()

        # camera and noise
        fx, fy, cx, cy = self.frontend.camera
        K = gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)
        uv_measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 30)
        landmark_prior_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        pose_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6)*0.1)

        # create projection factors
        for frame_id, curr_frame in self.frontend.frames.items():
            for landmark_id in curr_frame.observed_landmark_id:
                landmark = self.frontend.landmarks[landmark_id]
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

    def plot(self):
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
        estimated_traj = np.zeros([3, len(self.frontend.frames)])
        input_traj = np.zeros([3, len(self.frontend.frames)])
        for frame_id, curr_frame in self.frontend.frames.items():
            estimated_traj[:, frame_id] = self.current_estimate.atPose3(X(frame_id)).translation()
            input_traj[:, frame_id] = curr_frame.translation.flatten()
        ax.plot3D(estimated_traj[0,:], estimated_traj[1,:], estimated_traj[2,:], 'red', label='estimated')
        ax.plot3D(input_traj[0,:], input_traj[1,:], input_traj[2,:], 'blue', label='input')
        ax.legend()
        plt.show()
        # plt.pause(100)

if __name__ == '__main__':
    dataset_type = 'soulcity'
    dataset_folder = os.path.expanduser('~/Projects/curly_slam/data/soulcity')
    frontend = FrontendEvaluate(dataset_type, dataset_folder, start=0)
    frontend.load_dataset(os.path.expanduser('~/Projects/curly_slam/data/curly_frontend/curly.txt'))
    backend = GTSAM_Backend(frontend)
    backend.optimize()
    print("EOF")