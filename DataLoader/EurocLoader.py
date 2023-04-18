import os
import cv2
import glob
import yaml
import csv
import logging

from utils.PinholeCamera import PinholeCamera


class EurocLoader(object):
    default_config = {
        "sequence": "MH_01_easy",
        "start": 0,
        "camera_folder": 'cam0'
    }

    def __init__(self, config={}):
        self.config = self.default_config
        self.config = {**self.config, **config}
        logging.info("Euroc Dataset config: ")
        logging.info(self.config)
        
        cam_info_path = os.path.join(self.config["root_path"], self.config["sequence"], 'mav0', 
                                     self.config["camera_folder"], 'sensor.yaml')
        if(not os.path.exists(cam_info_path)):
            raise ValueError(f'Camera info file {cam_info_path} not found!')
        
        with open(cam_info_path, 'r') as f:
            cam_info = yaml.load(f, Loader=yaml.FullLoader)
            self.cam = PinholeCamera(cam_info['resolution'][0], cam_info['resolution'][1],
                                     cam_info['intrinsics'][0], cam_info['intrinsics'][1],
                                     cam_info['intrinsics'][2], cam_info['intrinsics'][3])
            
        metadata_path = os.path.join(self.config["root_path"], self.config["sequence"], 'mav0',
                                        self.config["camera_folder"], 'data.csv')
        if(not os.path.exists(metadata_path)):
            raise ValueError(f'Metadata file {metadata_path} not found!')
        
        #read csv
        self.all_ids = []
        with open(metadata_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            self.all_ids = [row[1] for row in reader if row[1] != 'filename']

        # image id
        self.img_id = self.config["start"]
        if(self.config["end"] > 0):
            self.img_N = self.config["end"]
        else:
            self.img_N = len(glob.glob(pathname=os.path.join(self.config["root_path"],self.config["sequence"],
                                                              'mav0', self.config["camera_folder"], 'data')+'/*.png'))

    def __getitem__(self, item):
        file_name = os.path.join(self.config["root_path"],self.config["sequence"],
                                'mav0', self.config["camera_folder"], 'data', self.all_ids[item])
        img = cv2.imread(file_name)
        return img

    def __iter__(self):
        return self

    def __next__(self):
        if self.img_id < self.img_N:
            file_name = os.path.join(self.config["root_path"],self.config["sequence"],
                                'mav0', self.config["camera_folder"], 'data', self.all_ids[self.img_id])
            img = cv2.imread(file_name)

            self.img_id += 1

            return img
        raise StopIteration()

    def __len__(self):
        return self.img_N - self.config["start"]


