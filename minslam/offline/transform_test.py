from frontend_evaluate import FrontendEvaluate

if __name__ == '__main__':
    dataset_type = 'tartanair'
    dataset_folder = os.path.expanduser(
        '~/Projects/curly_slam/data/tartanair/scenes/soulcity/Easy/P001')
    frontend_file = os.path.expanduser(
        '~/Projects/curly_slam/data/log/abandonedfactory_easy_p001.txt')
    dataset_path = {
        'depth': dataset_folder+'/depth_left',
        'color': dataset_folder+'/image_left',
        'frontend': frontend_file,
        'gt_traj': dataset_folder+'/pose_left.txt',
        'odom_traj': dataset_folder+'/pose_left.txt'
    }

    frontend = FrontendEvaluate(dataset_type, dataset_path)
    frontend.load_dataset(start=0, end=500)
    frontend.evaluate(viz_matches=False)
    frame_id1 = 0
    frame_id2 = 30
    frontend.frames
    print("EOF")