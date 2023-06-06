import numpy as np
from spatialmath import SE3
import cv2
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import gtsam
from gtsam.symbol_shorthand import L, X
from minslam.camera import PinholeCamera

@dataclass
class ProjectionFactor:
    global_id: int = -1
    smart_factor: gtsam.SmartProjectionPose3Factor = None
    generic_factors: list[gtsam.GenericProjectionFactorCal3_S2] = field(default_factory=list)
    measurements: dict[int, list[float]] = field(default_factory=dict)

class Backend():
    def __init__(self, params):
        # Parameters
        self.params = params
        self.camera = PinholeCamera(params)
        # factor
        smart_factor_params = gtsam.SmartProjectionParams()
        smart_factor_params.setDegeneracyMode(gtsam.DegeneracyMode.ZERO_ON_DEGENERACY)
        smart_factor_params.setRankTolerance(1)
        self.smart_factor_params = smart_factor_params
        # graph
        self.frame_id_list = []
        self.factors = {}
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.current_estimate = gtsam.Values()

    def add_keyframe(self, frame_id, pose, measurements):
        '''
        @param frame_id: current frame id
        @param pose: initial pose estimate
        @param measurements: [global_id, u, v, depth]
        '''

        # process data
        fx, fy, cx, cy = self.camera.camera_matrix
        K = gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)
        pose_gtsam = gtsam.Pose3(gtsam.Rot3(pose.R),gtsam.Point3(pose.t))

        # add initial estimate for pose
        self.initial_estimate.insert(X(frame_id), pose_gtsam)

        # add prior factor if it is the first keyframe
        if len(self.frame_id_list) == 0:
            sigma_r = self.params['backend']['pose_prior']['sigma_rotation']
            sigma_t = self.params['backend']['pose_prior']['sigma_translation']
            print('add prior factor for frame ', frame_id)
            pose_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([sigma_t]*3+[sigma_r]*3))
            self.graph.push_back(gtsam.PriorFactorPose3(
                X(frame_id), pose_gtsam, pose_prior_noise
            ))
        
        # add projection factor
        for measurement in measurements:
            global_id, u, v, depth = measurement

            # add projection factor
            if global_id not in self.factors:
                self.factors[global_id] = ProjectionFactor(global_id)

            # add measurement
            self.factors[global_id].measurements[frame_id] = [u, v, depth]

            # add smart factor
            if self.params['backend']['smart_projection_factor']['enabled']:
                smart_factor = self.factors[global_id].smart_factor
                if smart_factor is None:
                    noise = self.params['backend']['smart_projection_factor']['noise']
                    noise_model = gtsam.noiseModel.Isotropic.Sigma(2, noise)
                    smart_factor = gtsam.SmartProjectionPose3Factor(noise_model, K, self.smart_factor_params)
                    self.factors[global_id].smart_factor = smart_factor
                smart_factor.add(np.array([u, v]), X(frame_id))
                if smart_factor.size()>1:
                    # print('add smart factor to graph, frame=', frame_id, ', landmark=', global_id)
                    self.graph.push_back(smart_factor)
                    # point = self.camera.back_project(u,v, depth,pose)
                    # print('camera.back_project: ', point.flatten(), ', depth=',depth)
                    smart_factor.error(self.initial_estimate)
                    # smart_factor.print()

            # add generic factor
            if self.params['backend']['generic_projection_factor']['enabled']:
                generic_factors = self.factors[global_id].generic_factors
                global_position_xyz = gtsam.Point3(self.camera.back_project(u, v, depth, pose).flatten())
                if len(generic_factors) == 0:
                    self.initial_estimate.insert(L(global_id), global_position_xyz)
                noise = self.params['backend']['generic_projection_factor']['noise']
                noise_model = gtsam.noiseModel.Isotropic.Sigma(2, noise)
                noise_model_robust = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.345), noise_model)
                generic_factors.append(gtsam.GenericProjectionFactorCal3_S2(
                    np.array([u, v]), noise_model_robust, X(frame_id), L(global_id), K
                ))
                if len(generic_factors) == 2:
                    for generic_factor in generic_factors:
                        self.graph.push_back(generic_factor)
                elif len(generic_factors) > 2:
                    self.graph.push_back(generic_factors[-1])

        # save to frame id list
        self.frame_id_list.append(frame_id)

    def optimize(self, optimizer='ISAM2'):
        print('before optimizaiton error: ', self.graph.error(self.initial_estimate))
        if optimizer=='LM':
            optimizer_params = gtsam.LevenbergMarquardtParams()
            optimizer_params = gtsam.LevenbergMarquardtParams.CeresDefaults()
            optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimate, optimizer_params)
            current_estimate = optimizer.optimize()
        elif optimizer=='GN':
            optimizer = gtsam.GaussNewtonOptimizer(self.graph, self.initial_estimate)
            current_estimate = optimizer.optimize()
        elif optimizer=='ISAM2':
            optimization_params = gtsam.ISAM2DoglegParams()
            # optimization_params.setVerbose(True)
            optimizer_params = gtsam.ISAM2Params()
            optimizer_params.setRelinearizeThreshold(0.001)
            optimizer_params.setOptimizationParams(optimization_params)
            optimizer = gtsam.ISAM2(optimizer_params)
            optimizer.update(self.graph, self.initial_estimate)
            current_estimate = optimizer.calculateEstimate()
        print('after optimizaiton error: ', self.graph.error(current_estimate))

        self.current_estimate = current_estimate