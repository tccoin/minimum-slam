'''
playground for testing GTSAM
'''
import os
import gtsam
import numpy as np
from gtsam.symbol_shorthand import L, X
import matplotlib.pyplot as plt
from frontend_evaluate import FrontendEvaluate
import math
import matplotlib
# matplotlib.use('Agg')
class GTSAM_Backend():

    def __init__(self):
        # setup canvas
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        ax.view_init(23,-150)
        self.ax = ax

    def set_frontend(self, frontend):
        self.frontend = frontend

    def optimize(self, use_smart_factor=True):
        # graph
        graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()

        # factors
        smart_factors = {}
        smart_factor_params = gtsam.SmartProjectionParams()
        smart_factor_params.setDegeneracyMode(gtsam.DegeneracyMode.ZERO_ON_DEGENERACY) #HANDLE_INFINITY ZERO_ON_DEGENERACY
        smart_factor_params.setRankTolerance(1)
        # smart_factor_params.setLandmarkDistanceThreshold(100)
        # smart_factor_params.setDynamicOutlierRejectionThreshold(10000)

        # camera and noise
        fx, fy, cx, cy = self.frontend.camera
        K = gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)
        uv_measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1)
        landmark_prior_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1)
        first_pose_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6)*0.01)
        pose_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([20*math.pi/180]*3+[1]*3)
        )
        camera_pose = gtsam.Pose3(
            gtsam.Rot3(self.frontend.cameraPose.rotation),
            gtsam.Point3(self.frontend.cameraPose.translation.flatten())
        )

        # debug
        count = [0]*10
        first_frame_id = next(iter(self.frontend.frames))
        landmark_set = set()

        # create projection factors
        for frame_id, curr_frame in self.frontend.frames.items():
            print(f'=========== FRAME {frame_id} ===========')
            pose = gtsam.Pose3(
                gtsam.Rot3(curr_frame.rotation),
                gtsam.Point3(curr_frame.translation.flatten())
            )
            initial_estimate.insert(X(frame_id), pose)
            if frame_id==first_frame_id:
                graph.push_back(gtsam.PriorFactorPose3(
                    X(frame_id), pose, first_pose_prior_noise
                ))
            # else:
            #     graph.push_back(gtsam.PriorFactorPose3(
            #         X(frame_id), pose, pose_prior_noise
            #     ))
            for landmark_id in curr_frame.observed_landmark_id:
                landmark = self.frontend.landmarks[landmark_id]
                if use_smart_factor:
                    if landmark.frames[0]==frame_id:
                        factor = gtsam.SmartProjectionPose3Factor(uv_measurement_noise, K, camera_pose, smart_factor_params)
                        smart_factors[landmark_id] = factor
                    else:
                        factor = smart_factors[landmark_id]
                    factor.add(np.array(landmark.uv[frame_id]), X(frame_id))
                    if landmark.frames[-1]==frame_id:
                        # print(len(landmark.uv), factor.size())
                        try:
                            # factor.print()
                            error = factor.error(initial_estimate)
                        except:
                            print('Failed when calculating error')
                            continue
                        if factor.size()>=2:
                            count[0] += 1
                            gt_point = landmark.gt_xyz[frame_id].flatten()
                        if factor.size()>=2 and error==0:
                            # factor.print()
                            # factor.printPoint()
                            status = factor.getStatus()
                            if status!=2:
                                print(f'Triangulation failed. Status={status}')
                            count[1] += 1
                        if factor.size()>=2 and error>0:
                            estimated_point = factor.getPoint()
                            error_gt = np.linalg.norm(gt_point-estimated_point)
                            print('result: ',landmark_id,gt_point, estimated_point, error_gt, error/len(landmark.uv), len(landmark.uv))
                            count[2] += 1
                        if factor.size()>=2 and error>1000:
                            count[4] += 1
                        # if frame_id==45 and factor.size()>=2:
                        #     self.frontend.show_matches('test', list(landmark.uv.keys()), [list(landmark.uv.values())], -1)
                        if factor.size()>=2 and error>0 and error/len(landmark.uv)<2000:
                            # print(error_gt, error)
                            # factor.printPoint()
                            count[3] += 1
                            graph.push_back(factor)
                            for x in landmark.uv.keys():
                                landmark_set.add(x)
                            # factor.print()
                            # print(f'size: {factor.size()}, error: {error}, average error: {error/factor.size()}')
                            # print('==========')
                        # print(f'added!')
                else:
                    if landmark.frames[0]==frame_id:
                        initial_estimate.insert(L(landmark_id), gtsam.Point3(*landmark.xyz[frame_id]))
                        graph.push_back(gtsam.PriorFactorPoint3(
                            L(landmark_id), gtsam.Point3(*landmark.xyz[frame_id]), landmark_prior_noise
                        ))
                    graph.push_back(gtsam.GenericProjectionFactorCal3_S2(
                        np.array(landmark.uv[frame_id]), uv_measurement_noise, X(frame_id),
                        L(landmark_id), K, camera_pose
                    ))
        # for i in range(landmark.frames[-1]+1):
        #     if not i in landmark_set:
        #         print(f'frame {i} has no measurement')

        # debug
        print(f'count: {count}')

        # optimize
        print('graph.size(): ', graph.size())
        print('before optimizaitonerror(): ', graph.error(initial_estimate))

        optimizer_params = gtsam.LevenbergMarquardtParams()
        optimizer_params = gtsam.LevenbergMarquardtParams.CeresDefaults()
        # optimizer_params.setVerbosityLM('SUMMARY')
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, optimizer_params)
        # optimizer = gtsam.GaussNewtonOptimizer(graph, initial_estimate)
        self.current_estimate = optimizer.optimize()

        # optimization_params = gtsam.ISAM2DoglegParams()
        # optimization_params.setVerbose(True)
        # optimizer_params = gtsam.ISAM2Params()
        # optimizer_params.setRelinearizeThreshold(0.01)
        # optimizer_params.setOptimizationParams(optimization_params)
        # optimizer = gtsam.ISAM2(optimizer_params)
        # optimizer.update(graph, initial_estimate)
        # self.current_estimate = optimizer.calculateEstimate()

        
        print('after optimizaitonerror(): ', graph.error(self.current_estimate))

        # try:
        #     self.current_estimate = optimizer.optimize()
        # except:
        #     print('Optimization failed')

        # plot
        self.plot()

    def plot(self, plot_input_traj=True, plot_estimated_traj=True, plot_gt_traj=True):
        ax = self.ax
        ax.clear()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        # ax.set_xlim3d(-10, 10)
        # ax.set_ylim3d(-10, 10)
        # ax.set_zlim3d(-10, 10)
        first_id = next(iter(self.frontend.frames))

        # plot input_traj
        if plot_input_traj:
            traj = np.zeros([3, len(self.frontend.frames)])
            for frame_id, curr_frame in self.frontend.frames.items():
                offset = frame_id-first_id
                traj[:, offset] = curr_frame.translation.flatten()
            ax.plot3D(traj[0,:], traj[1,:], traj[2,:]+0.01, 'blue', label='input')

        # plot gt_traj
        if plot_gt_traj:
            traj = np.zeros([3, len(self.frontend.frames)])
            for frame_id, curr_frame in self.frontend.frames.items():
                offset = frame_id-first_id
                traj[:, offset] = curr_frame.gt_translation.flatten()
            ax.plot3D(traj[0,:], traj[1,:], traj[2,:], 'orange', label='ground truth')

        # plot estimated_traj
        if plot_estimated_traj:
            traj = np.zeros([3, len(self.frontend.frames)])
            for frame_id, curr_frame in self.frontend.frames.items():
                offset = frame_id-first_id
                traj[:, offset] = self.current_estimate.atPose3(X(frame_id)).translation()
                # ax.text(traj[0,offset], traj[1,offset], traj[2,offset], frame_id)
            ax.plot3D(traj[0,:], traj[1,:], traj[2,:], 'red', label='estimated')

        # show the plot
        ax.legend()
        plt.show(
            # block=False
        )
        plt.pause(0.1)

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
    backend = GTSAM_Backend()
    for i in range(250,600,1):
        print(f'===== FRAME {i} =====')
        frontend.load_dataset(start=0, end=i)
        frontend.evaluate(viz_matches=False)
        backend.set_frontend(frontend)
        backend.optimize()
    print("EOF")