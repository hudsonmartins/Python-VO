import os
import sys
import numpy as np
import cv2
import argparse
import yaml
import logging
from scipy.spatial.transform import Rotation
from DataLoader import create_dataloader
from Detectors import create_detector
from Matchers import create_matcher
from VO.VisualOdometry import VisualOdometry, AbosluteScaleComputer
from utils.tools import plot_keypoints, plot_matches

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
    last_img = None
    matches_img = None
    for i, img  in enumerate(loader):
        print(i,'/', len(loader))
        if(args.skip_frames == 0 or i%args.skip_frames == 0):           
            
            R, t, kptdescs, matches = vo.update(img, 1)
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
                if(last_img is not None and matches is not None):
                    matches_img = plot_matches(last_img, img, 
                                            matches['cur_keypoints'],
                                            matches['ref_keypoints'], 
                                            matches['match_score'])
                kpts_img = plot_keypoints(img, kptdescs["cur"]["keypoints"])

                kpts_dir = os.path.join(args.output_path, 'kpts', fname.split('.')[0])
                matches_dir = os.path.join(args.output_path, 'matches', fname.split('.')[0])
                os.makedirs(kpts_dir, exist_ok=True)
                os.makedirs(matches_dir, exist_ok=True)
                cv2.imwrite(os.path.join(kpts_dir, str(i)+'.png'), kpts_img)
                if(matches_img is not None):
                    cv2.imwrite(os.path.join(matches_dir, str(i)+'.png'), matches_img)
            
           
            if(args.save_raw_outputs):
                """
                Raw outputs: 
                   Keypoints (N, 2) -- N Keypoints coordinates inside the image (x, y)
                   Descriptors (N, D): N Descriptions vector with D dimensions
                   Matches (M, 3) -- Index of the M matched keypoints in each image and 
                                        the matching score (kpt0_id, kpt1_id, score)
                """
                raw_outputs_dir = os.path.join(args.output_path, 'raw_outputs', fname.split('.')[0])
                os.makedirs(raw_outputs_dir, exist_ok=True)
                np.save(os.path.join(raw_outputs_dir, str(i)+'_kpts.npy'), kptdescs["cur"]["keypoints"])
                np.save(os.path.join(raw_outputs_dir, str(i)+'_desc.npy'), kptdescs["cur"]["descriptors"])

                if(matches is not None):
                    matches_scores = np.concatenate((matches['cur_keypoints'], 
                                                     matches['match_score'].reshape(-1,1)), axis=1)
                    np.save(os.path.join(raw_outputs_dir, str(i-1)+'_'+str(i)+'_matches.npy'), matches_scores)
            
            last_img = img


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
    parser.add_argument('--save_raw_outputs', action='store_true', help='Save raw outputs of the detector and matcher')

    args = parser.parse_args()

    logging.basicConfig(level=logging._nameToLevel[args.logging])

    run(args)
