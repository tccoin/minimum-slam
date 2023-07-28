import plotly.graph_objects as go
import numpy as np
import gtsam


from minslam.data_loader import TartanAirLoader, plot_trajectory
from minslam.frontend import Frontend
from minslam.params import Params
from minslam.backend import Backend
from minslam.camera import PinholeCamera


# plot the results
fig = go.Figure()
# plot_trajectory(odom_traj[:n_keyframes], 'odom', fig)
# plot_trajectory(gt_traj[:n_keyframes], 'gt', fig)
# plot_trajectory(estimated_traj[:n_keyframes], 'estimated', fig)
fig.show()


# define frontend
def run_frontend_once(frontend):
    pose = dataset.read_current_ground_truth()
    while not frontend.keyframe_selection(pose):
        if not dataset.load_next_frame():
            break
        pose = dataset.read_current_ground_truth()
    color, depth = dataset.read_current_rgbd()
    frontend.add_keyframe(pose, color, depth, dataset.curr_index)
    print(f'--- Added keyframe {frontend.frame_id} (seq id: {dataset.curr_index}) ---')
    more_points_n = params['frontend']['feature']['number']
    frontend.extract_features(more_points_n, append_mode=False)
    print('extracting features:', len(frontend.curr_frame.keypoints), f'(expected {more_points_n})')
    if frontend.frame_id > 0:
        frontend.match_features()
        print('matching features:', len(frontend.curr_frame.matches))
        frontend.eliminate_outliers()
    frontend.assign_global_id()
    return frontend.curr_frame

# load the dataset
params = Params('../params/tartanair.yaml')
frontend = Frontend(params)
backend = Backend(params)
dataset = TartanAirLoader('../data/P006/')
start_index = 0
dataset.set_curr_index(start_index)
n_keyframes = 10

# read the ground truth and odometry
dataset.load_ground_truth()
gt_poses = dataset.gt
odom_poses = dataset.add_noise(gt_poses, [1e-4, 3e-4], [1e-3, 1e-3], seed=100, start=start_index)
dataset.set_odometry(odom_poses)

# for plotting
gt_traj = np.zeros((n_keyframes, 3))
odom_traj = np.zeros((n_keyframes, 3))

# run the frontend and backend
for i in range(n_keyframes):
    # get results from frontend
    frontend_frame = run_frontend_once(frontend)
    gt_traj[i] = dataset.read_current_ground_truth().t
    odom_traj[i] = dataset.read_current_odometry().t
    measurements = []
    frame_id = frontend_frame.frame_id
    for landmark in frontend_frame.landmarks:
        global_id = landmark.global_id
        measurement = landmark.measurements[frame_id] # u, v, depth
        measurements.append((global_id, *measurement))
    print(f'add {len(measurements)} measurements to backend')
    # add measurements to backend
    backend.add_keyframe(frame_id, frontend_frame.odom_pose, measurements)
# optimize the backend
backend.optimize(optimizer='LM')
backend_estimate = backend.current_estimate
# estimated_traj = gtsam.utilities.extractPose3(backend_estimate)[:, -3:]
