from pathlib import Path
from utils.tools import *
import logging
from Matchers.superglue.superglue_patches import SuperGluePatches
import torch.nn.functional as F
torch.set_grad_enabled(False)

class SuperGluePatchesMatcher(object):
    default_config = {
        "descriptor_dim": 256,
        "weights": "outdoor",
        "keypoint_encoder": [32, 64, 128, 256],
        "GNN_layers": ["self", "cross"] * 9,
        "sinkhorn_iterations": 100,
        "match_threshold": 0.2,
        "cuda": True,
        "patch_size": 10
    }

    def __init__(self, config={}):
        self.config = self.default_config
        self.config = {**self.config, **config}
        logging.info("SuperGlue matcher config: ")
        logging.info(self.config)

        self.device = 'cuda' if torch.cuda.is_available() and self.config["cuda"] else 'cpu'

        assert self.config['weights'] in ['indoor', 'outdoor']
        path = Path(__file__).parent
        #path = path / 'superglue/superglue_{}.pth'.format(self.config['weights'])
        path = path / 'superglue/weights/superglue_patches.pth'
        self.config["path"] = path

        logging.info("creating SuperGlue matcher...")
        self.superglue = SuperGluePatches(self.config).to(self.device)

    def __call__(self, kptdescs, images=None):
        # setup data for superglue
        logging.debug("prepare input data for superglue...")
        data = {}
        data['image_size0'] = torch.from_numpy(kptdescs["ref"]["image_size"]).float().to(self.device)
        data['image_size1'] = torch.from_numpy(kptdescs["cur"]["image_size"]).float().to(self.device)
            
        if "torch" in kptdescs["cur"]:
            data['scores0'] = kptdescs["ref"]["torch"]["scores"][0]#.float().to(self.device).unsqueeze(0)
            data['keypoints0'] = kptdescs["ref"]["torch"]["keypoints"][0]#.float().to(self.device).unsqueeze(0)
            data['descriptors0'] = kptdescs["ref"]["torch"]["descriptors"][0]#.float().to(self.device).unsqueeze(0)

            data['scores1'] = kptdescs["cur"]["torch"]["scores"][0]#.float().to(self.device).unsqueeze(0)
            data['keypoints1'] = kptdescs["cur"]["torch"]["keypoints"][0]#.float().to(self.device).unsqueeze(0)
            data['descriptors1'] = kptdescs["cur"]["torch"]["descriptors"][0]#.float().to(self.device).unsqueeze(0)
            

        else:
            data['scores0'] = torch.from_numpy(kptdescs["ref"]["scores"])#.float().to(self.device).unsqueeze(0)
            data['keypoints0'] = torch.from_numpy(kptdescs["ref"]["keypoints"])#.float().to(self.device).unsqueeze(0)
            data['descriptors0'] = torch.from_numpy(kptdescs["ref"]["descriptors"])#.float().to(self.device).unsqueeze(0).transpose(1, 2)

            data['scores1'] = torch.from_numpy(kptdescs["cur"]["scores"])#.float().to(self.device).unsqueeze(0)
            data['keypoints1'] = torch.from_numpy(kptdescs["cur"]["keypoints"])#.float().to(self.device).unsqueeze(0)
            data['descriptors1'] = torch.from_numpy(kptdescs["cur"]["descriptors"])#.float().to(self.device).unsqueeze(0).transpose(1, 2)
        data = pad_data(data, 400, self.device)
        if(images != None):
            images[0] = resize_image(images[0])
            images[1] = resize_image(images[1])
            img0, img1 = torch.Tensor(images[0]).to(self.device), torch.Tensor(images[1]).to(self.device)
            patches_pts0 = get_patches_points(data['keypoints0'])
            patches_pts1 = get_patches_points(data['keypoints1'])
            img0 = img0.permute(2, 0, 1)
            img1 = img1.permute(2, 0, 1)
            img0 = torch.unsqueeze(img0, dim=0)
            img1 = torch.unsqueeze(img1, dim=0)

            img0_pad = F.pad(img0, (self.config['patch_size']//2, 
                                    self.config['patch_size']//2, 
                                    self.config['patch_size']//2, 
                                    self.config['patch_size']//2), "reflect")
            img1_pad = F.pad(img1, (self.config['patch_size']//2, 
                                    self.config['patch_size']//2, 
                                    self.config['patch_size']//2, 
                                    self.config['patch_size']//2), "reflect")
            print('img0_pad ', img0_pad.shape)
            print('patches_pts0 ', patches_pts0.shape)
            patches0 = get_pixels_from_pts(img0_pad, patches_pts0, output_shape=(patches_pts0.shape[1], patches_pts0.shape[2], self.config['patch_size']//2, self.config['patch_size']//2, 3))
            patches1 = get_pixels_from_pts(img1_pad, patches_pts1, output_shape=(patches_pts1.shape[1], patches_pts1.shape[2], self.config['patch_size']//2, self.config['patch_size']//2, 3))
                
            patches = {'patches0': patches0.to(self.device), 'patches1': patches1.to(self.device)}
            data = {**data, **patches}

        # Forward
        logging.debug("matching keypoints with superglue...")
        pred = self.superglue(data)

        # get matching keypoints
        kpts0 = kptdescs["ref"]["keypoints"]
        kpts1 = kptdescs["cur"]["keypoints"]

        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().detach().numpy()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Sort them in the order of their confidence.
        match_conf = []
        for i, (m, c) in enumerate(zip(matches, confidence)):
            match_conf.append([i, m, c])
        match_conf = sorted(match_conf, key=lambda x: x[2], reverse=True)

        valid = [[l[0], l[1]] for l in match_conf if l[1] > -1]
        v0 = [l[0] for l in valid]
        v1 = [l[1] for l in valid]
        mkpts0 = kpts0[v0]
        mkpts1 = kpts1[v1]

        ret_dict = {
            "ref_keypoints": mkpts0,
            "cur_keypoints": mkpts1,
            "match_score": confidence[v0]
        }

        return ret_dict


if __name__ == "__main__":
    from DataLoader.SequenceImageLoader import SequenceImageLoader
    from DataLoader.KITTILoader import KITTILoader
    from Detectors.SuperPointDetector import SuperPointDetector

    loader = SequenceImageLoader()
    detector = SuperPointDetector({"cuda": 0})
    matcher = SuperGlueMatcher({"cuda": 0, "weights": "indoor"})

    kptdescs = {}
    imgs = {}
    for i, img in enumerate(loader):
        imgs["cur"] = img
        kptdescs["cur"] = detector(img)
        if i >= 1:
            matches = matcher(kptdescs)
            img = plot_matches(imgs['ref'], imgs['cur'],
                               matches['ref_keypoints'][0:200], matches['cur_keypoints'][0:200],
                               matches['match_score'][0:200], layout='lr')
            cv2.imshow("track", img)
            if cv2.waitKey() == 27:
                break

        kptdescs["ref"], imgs["ref"] = kptdescs["cur"], imgs["cur"]
