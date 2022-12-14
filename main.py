import os
import sys
import numpy as np
import cv2
import argparse
import yaml
import logging
from utils.tools import resize_image
from DataLoader import create_dataloader
from Detectors import create_detector
from Matchers import create_matcher
from VO.VisualOdometry import VisualOdometry, AbosluteScaleComputer
from scipy.spatial.transform import Rotation

def run(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config['dataset']['root_path'] = args.root_path
    config['dataset']['camera_folder'] = args.camera_folder
    # create dataloader
    loader = create_dataloader(config['dataset'])
    # create detector
    detector = create_detector(config['detector'])
    # create matcher
    matcher = create_matcher(config['matcher'])

    # log
    fname = ''
    if(args.traj_file_name == None):
        fname = args.config.split('/')[-1].split('.')[0]
        fname += '_'+config['dataset']['sequence']+'_'+config['dataset']['camera_folder']+'_skip_'+str(args.skip_frames)+'.txt'
    else:
        fname = args.traj_file_name

    if(args.timestamps_file != None):
        timestamps_file = open(args.timestamps_file , 'rb')
        timestamps = timestamps_file.readlines()
        timestamps = [float(i) for i in timestamps]
    
    os.makedirs(args.output_path, exist_ok=True)
    log_fopen = open(os.path.join(args.output_path, fname), mode='a')
    vo = VisualOdometry(detector, matcher, loader.cam)

    for i, img  in enumerate(loader):
        print(i,'/', len(loader))
        if(args.skip_frames == 0 or i%args.skip_frames == 0):           
            
            R, t, viz = vo.update(img, 1, args.save_viz)
            if(args.save_format == 'kitti'):
                print(R[0, 0], R[0, 1], R[0, 2], t[0, 0],
                    R[1, 0], R[1, 1], R[1, 2], t[1, 0],
                    R[2, 0], R[2, 1], R[2, 2], t[2, 0],
                    file=log_fopen)
            elif(args.save_format == 'tum'):
                r = Rotation.from_matrix([[R[0, 0], R[0, 1], R[0, 2]],
                                        [R[1, 0], R[1, 1], R[1, 2]],
                                        [R[2, 0], R[2, 1], R[2, 2]]])
                q = r.as_quat()
                print(str(timestamps[i]), t[0, 0], t[1, 0], t[2, 0], q[0], q[1], q[2], q[3],
                    file=log_fopen)
            
            if(args.save_viz):
                kpts_dir = os.path.join(args.output_path, 'kpts', fname.split('.')[0])
                matches_dir = os.path.join(args.output_path, 'matches', fname.split('.')[0])
                os.makedirs(kpts_dir, exist_ok=True)
                os.makedirs(matches_dir, exist_ok=True)
                if(len(viz[0]) > 0):
                    cv2.imwrite(os.path.join(kpts_dir, str(i)+'.png'), viz[0])
                if(len(viz[1]) > 0):
                    cv2.imwrite(os.path.join(matches_dir, str(i)+'.png'), viz[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='python_vo')
    parser.add_argument('--root_path', type=str, default='datasets/kitti', help='path to the dataset')
    parser.add_argument('--config', type=str, default='params/kitti_superpoint_supergluematch.yaml',
                        help='config file')
    parser.add_argument('--logging', type=str, default='INFO',
                        help='logging level: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL')
    parser.add_argument('--camera_folder', type=str, default='image_0', help='folder to desired camera')
    parser.add_argument('--skip_frames', type=int, default=0, help='Number of frames to skip before estimating motion')
    parser.add_argument('--save_format', type=str, default='kitti', help='Format to save data, if tum the timestamps file is needed')
    parser.add_argument('--output_path', type=str, default='results', help='Path to save the results')
    parser.add_argument('--traj_file_name', type=str, default=None, help='Name of the trajectory file that will be saved in the output_path')
    parser.add_argument('--timestamps_file', type=str, default=None, help='Path to the timestamps file if using tum format')
    parser.add_argument('--image_size', type=int, default=720)
    parser.add_argument('--save_viz', action='store_true')

    args = parser.parse_args()

    logging.basicConfig(level=logging._nameToLevel[args.logging])

    run(args)
