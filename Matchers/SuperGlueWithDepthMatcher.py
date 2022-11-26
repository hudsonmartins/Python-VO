from pathlib import Path
from utils.tools import *
import logging
from Matchers.superglue.supergluewithdepth import SuperGlueWithDepth
from Matchers.superglue.monodepth2 import MonoDepth

torch.set_grad_enabled(False)

class SuperGlueWithDepthMatcher(object):
    default_config = {
        "descriptor_dim": 256,
        "weights": "outdoor",
        "keypoint_encoder": [32, 64, 128, 256],
        "GNN_layers": ["self", "cross"] * 9,
        "sinkhorn_iterations": 100,
        "match_threshold": 0.2,
        "cuda": True
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
        path = path / 'superglue/model_47.pth'
        self.config["path"] = path

        logging.info("creating SuperGlue matcher...")
        self.superglue = SuperGlueWithDepth(self.config).to(self.device)
        self.mono_depth = MonoDepth().to(self.device).eval()

    def __call__(self, kptdescs, images=None):
        # setup data for superglue
        logging.debug("prepare input data for superglue...")
        data = {}
        data['image_size0'] = torch.from_numpy(kptdescs["ref"]["image_size"]).float().to(self.device)
        data['image_size1'] = torch.from_numpy(kptdescs["cur"]["image_size"]).float().to(self.device)
            
        if "torch" in kptdescs["cur"]:
            data['scores0'] = kptdescs["ref"]["torch"]["scores"][0].float().to(self.device).unsqueeze(0)
            data['keypoints0'] = kptdescs["ref"]["torch"]["keypoints"][0].float().to(self.device).unsqueeze(0)
            data['descriptors0'] = kptdescs["ref"]["torch"]["descriptors"][0].float().to(self.device).unsqueeze(0)

            data['scores1'] = kptdescs["cur"]["torch"]["scores"][0].float().to(self.device).unsqueeze(0)
            data['keypoints1'] = kptdescs["cur"]["torch"]["keypoints"][0].float().to(self.device).unsqueeze(0)
            data['descriptors1'] = kptdescs["cur"]["torch"]["descriptors"][0].float().to(self.device).unsqueeze(0)
            

        else:
            data['scores0'] = torch.from_numpy(kptdescs["ref"]["scores"]).float().to(self.device).unsqueeze(0)
            data['keypoints0'] = torch.from_numpy(kptdescs["ref"]["keypoints"]).float().to(self.device).unsqueeze(0)
            data['descriptors0'] = torch.from_numpy(kptdescs["ref"]["descriptors"]).float().to(self.device).unsqueeze(
                0).transpose(1, 2)

            data['scores1'] = torch.from_numpy(kptdescs["cur"]["scores"]).float().to(self.device).unsqueeze(0)
            data['keypoints1'] = torch.from_numpy(kptdescs["cur"]["keypoints"]).float().to(self.device).unsqueeze(0)
            data['descriptors1'] = torch.from_numpy(kptdescs["cur"]["descriptors"]).float().to(self.device).unsqueeze(
                0).transpose(1, 2)
        if(images != None):
            img0, img1 = torch.Tensor(images[0]).to(self.device), torch.Tensor(images[1]).to(self.device)
            d0 = self.mono_depth(img0)
            d1 = self.mono_depth(img1)
            d0 = get_kpts_depths(data['keypoints0'], d0)
            d1 = get_kpts_depths(data['keypoints1'], d1)
            disps = {'disp0': d0, 'disp1': d1}
            data = {**data, **disps}

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
