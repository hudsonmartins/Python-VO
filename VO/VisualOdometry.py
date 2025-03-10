# based on: https://github.com/uoip/monoVO-python

import numpy as np
import cv2
from skimage import io
class VisualOdometry(object):
    """
    A simple frame by frame visual odometry
    """

    def __init__(self, detector, matcher, cam):
        """
        :param detector: a feature detector can detect keypoints their descriptors
        :param matcher: a keypoints matcher matching keypoints between two frames
        :param cam: camera parameters
        :param first_frame: number of the initial frame
        """
        # feature detector and keypoints matcher
        self.detector = detector
        self.matcher = matcher

        # camera parameters
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)

        # frame index counter
        self.index = 0

        # keypoints and descriptors
        self.kptdescs = {}

        # pose of current frame
        self.cur_R = None
        self.cur_t = None

    def update(self, image, absolute_scale=1):
        """
        update a new image to visual odometry, and compute the pose
        :param image: input image
        :param absolute_scale: the absolute scale between current frame and last frame
        :return: R and t of current frame
        """

        kptdesc = self.detector(image)
        matches = None
        # first frame
        if self.index == 0:
            # save keypoints and descriptors
            self.kptdescs["cur"] = kptdesc
            self.img0 = image
            # start point
            self.cur_R = np.identity(3)
            self.cur_t = np.zeros((3, 1))
        else:
            # update keypoints and descriptors
            self.kptdescs["cur"] = kptdesc
            self.img0 = image
            # match keypoints
            matches = self.matcher(self.kptdescs)
                        
            # compute relative R,t between ref and cur frame
            E, mask = cv2.findEssentialMat(matches['cur_keypoints'], matches['ref_keypoints'],
                                           focal=self.focal, pp=self.pp,
                                           method=cv2.RANSAC, prob=0.999, threshold=1.0)

            _, R, t, mask = cv2.recoverPose(E, matches['cur_keypoints'], matches['ref_keypoints'],
                                            focal=self.focal, pp=self.pp)

            # get absolute pose based on absolute_scale
            if(absolute_scale > 0.1):
                self.cur_t = self.cur_t + absolute_scale * self.cur_R.dot(t)
                self.cur_R = R.dot(self.cur_R)
        
        self.kptdescs["ref"] = self.kptdescs["cur"]
        self.img1 = self.img0
        self.index += 1
        return self.cur_R, self.cur_t, self.kptdescs, matches #[kpts_img, matches_img]


class AbosluteScaleComputer(object):
    def __init__(self):
        self.prev_pose = None
        self.cur_pose = None
        self.count = 0

    def update(self, pose):
        self.cur_pose = pose

        scale = 1.0
        if self.count != 0:
            scale = np.sqrt(
                (self.cur_pose[0, 3] - self.prev_pose[0, 3]) * (self.cur_pose[0, 3] - self.prev_pose[0, 3])
                + (self.cur_pose[1, 3] - self.prev_pose[1, 3]) * (self.cur_pose[1, 3] - self.prev_pose[1, 3])
                + (self.cur_pose[2, 3] - self.prev_pose[2, 3]) * (self.cur_pose[2, 3] - self.prev_pose[2, 3]))

        self.count += 1
        self.prev_pose = self.cur_pose
        return scale


if __name__ == "__main__":
    from DataLoader.KITTILoader import KITTILoader
    from Detectors.HandcraftDetector import HandcraftDetector
    from Matchers.FrameByFrameMatcher import FrameByFrameMatcher

    loader = KITTILoader()
    detector = HandcraftDetector({"type": "SIFT"})
    matcher = FrameByFrameMatcher({"type": "FLANN"})
    absscale = AbosluteScaleComputer()

    vo = VisualOdometry(detector, matcher, loader.cam)
    for i, img in enumerate(loader):
        gt_pose = loader.get_cur_pose()
        R, t = vo.update(img, absscale.update(gt_pose))
