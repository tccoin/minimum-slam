import unittest
import minslam.data_loader as data_loader
import os


class TestDataLoader(unittest.TestCase):

    def test_add_or_ignore_timestamp(self):
        # print(sys.path.insert(0, ))
        dataset_folder = os.path.dirname(__file__)+'/traj_examples'
        loader = data_loader.DataLoader(dataset_folder)
        traj = loader._DataLoader__load_traj(
            'tum', 'tum_no_timestamp.txt', add_timestamps=True)
        self.assertEqual(traj.num_poses, 434)
        traj = loader._DataLoader__load_traj(
            'kitti', 'kitti_with_timestamp.txt', ignore_timestamps=True)
        self.assertEqual(traj.num_poses, 319)


if __name__ == '__main__':
    unittest.main()
