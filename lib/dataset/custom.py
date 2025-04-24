from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from glob import glob
import os.path as osp
import numpy as np
import json_tricks as json
import pickle
import logging
import cv2
import copy

from dataset.JointsDataset import JointsDataset

logger = logging.getLogger(__name__)

VAL_LIST = [
    # 'Take_035',
    # 'Take_036'
    # 'Take_019',
    # 'Take_020',
    # 'Take_022',
    # 'Take_023',
    'Take_024'
]

custom_joints_def = {
    'neck': 0,
    'nose': 1,
    'mid-hip': 2,
    'l-shoulder': 3,
    'l-elbow': 4,
    'l-wrist': 5,
    'l-hip': 6,
    'l-knee': 7,
    'l-ankle': 8,
    'r-shoulder': 9,
    'r-elbow': 10,
    'r-wrist': 11,
    'r-hip': 12,
    'r-knee': 13,
    'r-ankle': 14,
}

class Custom(JointsDataset):
    def __init__(self, cfg, is_train=True, transform=None):
        super().__init__(cfg, is_train, transform)
        
        if is_train:
            raise NotImplementedError("Training is not implemented for custom dataset")
        else:
            self.image_set = 'validation'
            self.sequence_list = VAL_LIST
            self._interval = 1

        self.num_joints = len(custom_joints_def)
        self.num_views = cfg.DATASET.CAMERA_NUM
        self.root_id = cfg.DATASET.ROOT_JOINT_ID
        
        self.has_evaluate_function = False
        self.transform = transform
        # self.cam_list = ['44310001', '44310002', '44310006', '44310010']
        self.cam_list = ['44310029', '44310042', '44310043', '44310048']
        
        self.cameras = self._get_cam()
        self.db_file = '{}_meta.pkl'.format(self.image_set, self.num_views)
        self.db_file = osp.join(self.dataset_dir, self.db_file)
        
        if osp.exists(self.db_file):
            info = pickle.load(open(self.db_file, 'rb'))
            assert info['sequence_list'] == self.sequence_list
            assert info['interval'] == self._interval
            self.db = info['db']
        else:
            self._get_db()
            info = {
                'sequence_list': self.sequence_list,
                'interval': self._interval,
                'db': self.db
            }
            pickle.dump(info, open(self.db_file, 'wb'))
        self.db_size = len(self.db)
    
    def _get_db(self):
        for seq in self.sequence_list:
            for i, image_path in enumerate(sorted(glob(osp.join(self.dataset_dir, seq, self.cam_list[0],'*.png')))):
                if i % self._interval != 0:
                    continue
                all_image_path = []
                missing_image = False
                for cam in self.cam_list:
                    all_image_path.append(osp.join(self.dataset_dir, seq, cam, osp.basename(image_path)))
                    if not osp.exists(all_image_path[-1]):
                        logger.warning("Image not found: {}. Skipped.".format(all_image_path[-1]))
                        missing_image = True
                        break

                if missing_image:
                    continue
                
                self.db.append({
                    'seq': seq,
                    'all_image_path': all_image_path
                })
        
        super()._rebuild_db()
        logger.info("=> {} images from {} views loaded".format(len(self.db), self.num_views))
        return
        
    def _get_cam(self):
        cameras = dict()
        
        for seq in self.sequence_list:
            cameras[seq] = []

            cam_file = osp.join(self.dataset_dir, seq, "calibration.json")
            with open(cam_file, "r") as f:
                calib = json.load(f)
            
            for cam in calib.keys():
                cam_param = {}
                cam_param['fx'] = calib[cam]['k'][0]
                cam_param['fy'] = calib[cam]['k'][1]
                cam_param['cx'] = calib[cam]['k'][2]
                cam_param['cy'] = calib[cam]['k'][3]
                cam_param['k'] = np.array([calib[cam]['d'][0], calib[cam]['d'][1], calib[cam]['d'][4]]).reshape(3, 1)
                cam_param['p'] = np.array([calib[cam]['d'][2], calib[cam]['d'][3]]).reshape(2, 1)
                proj_mat = np.array(calib[cam]['p']).reshape(3, 4)
                K = np.array(
                    [
                        [cam_param['fx'], 0, cam_param['cx']],
                        [0, cam_param['fy'], cam_param['cy']],
                        [0, 0, 1]
                    ]
                )
                
                T_cam_world = np.linalg.inv(K).dot(proj_mat)
                R = T_cam_world[:3, :3]
                t = T_cam_world[:3, 3].reshape(3, 1)
                cam_param['R'] = R # 世界坐标系到相机坐标系的旋转矩阵
                cam_param['T'] = -np.dot(R.T, t) # 相机在世界坐标系中的位置(mm)
                cameras[seq].append(cam_param)
        return cameras
    
    def __getitem__(self, idx):
        input, target, meta, input_heatmap = super().__getitem__(idx)
        return input, target, meta, input_heatmap
    
    def __len__(self):
        return self.db_size
    