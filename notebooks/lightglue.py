import torch
import numpy as np
import cv2
from minslam.data_loader import TartanAirLoader, plot_trajectory
from minslam.frontend import Frontend
from minslam.params import Params
from minslam.backend import Backend
from minslam import lightglue, viz2d

import plotly.graph_objects as go
import plotly.express as px

# load the dataset
params = Params('./params/lightglue_tartanair.yaml')
frontend = Frontend(params)
backend = Backend(params)
dataset = TartanAirLoader('./data/P006/')
start_index = 100
dataset.set_curr_index(start_index)
n_keyframes = 10

# read the ground truth and odometry
dataset.load_ground_truth()
gt_poses = dataset.gt
odom_poses = dataset.add_noise(gt_poses, [1e-4, 3e-4], [1e-3, 1e-3], seed=100, start=start_index)
dataset.set_odometry(odom_poses)

# use gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 'mps', 'cpu'

# set extractor and matcher
extractor = lightglue.SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher = lightglue.LightGlue(features='superpoint').eval().to(device)

def extract_features(frontend, more_points_n, append_mode=False):
    image = lightglue.utils.numpy_image_to_torch(frontend.curr_frame.color) 
    feats0 = extractor.extract(image.to(device))
    del image
    frontend.curr_frame.points = [p for p in feats0['keypoints'].cpu().numpy()[0]]
    frontend.curr_frame.keypoints = feats0

def match_features(frontend):
    feats0 = frontend.last_frame.keypoints
    feats1 = frontend.curr_frame.keypoints
    matches = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches = [lightglue.utils.rbd(x) for x in [feats0, feats1, matches]]
    frontend.curr_frame.matches = [cv2.DMatch(m[0], m[1], 0) for m in matches['matches'].cpu().numpy()]
for i in range(n_keyframes):
    print(f'--- Loading frame {i} ---')
    pose = dataset.read_current_ground_truth()
    while not frontend.keyframe_selection(pose):
        if not dataset.load_next_frame():
            break
        pose = dataset.read_current_ground_truth()
    color, depth = dataset.read_current_rgbd()
    frontend.add_keyframe(pose, color, depth, dataset.curr_index)
    print(f'--- Added keyframe {frontend.frame_id} (seq id: {dataset.curr_index}) ---')
    more_points_n = params['frontend']['feature']['number']
    extract_features(frontend, more_points_n, append_mode=False)
    print('extracting features:', len(frontend.curr_frame.keypoints), f'(expected {more_points_n})')
    if frontend.frame_id > 0:
        match_features(frontend)
        print('matching features:', len(frontend.curr_frame.matches))
        frontend.eliminate_outliers()
    frontend.assign_global_id()

fig = frontend.plot_matches()
fig.show()