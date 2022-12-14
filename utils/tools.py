import cv2
import numpy as np
import torch
import collections
import matplotlib.cm as cm
from skimage.transform import resize
import torch.nn.functional as F

def image2tensor(frame, device):
    return torch.from_numpy(frame / 255.).float()[None, None].to(device)


def get_kpts_depths(kpts, depths):
    kpts_depths = []
    i = kpts[0, :, 1].long()
    j = kpts[0, :, 0].long()
    kpts_depths = depths[:, 0, i, j]
    return kpts_depths
    

def patch_meshgrid(x_min, x_max, y_min, y_max):
    xs = torch.range(x_min, x_max-1, step=1)
    ys = torch.range(y_min, y_max-1, step=1)
    gx, gy = torch.meshgrid(xs, ys)
    grid = torch.cat((torch.unsqueeze(gx, dim=2), torch.unsqueeze(gy, dim=2)), dim=2)
    return grid


def get_patches_points(kpts, patch_size=10):
    all_pts_batch = None
    batch_size = kpts.size(0)

    for idx_in_batch in range(batch_size):
        patch_pts = None
        for pt in kpts[idx_in_batch]:
            pt = pt.long()
            coords = patch_meshgrid(pt[0], pt[0]+patch_size, pt[1], pt[1]+patch_size)
            coords = torch.reshape(coords, (patch_size**2, 2))
            if(patch_pts == None):
                patch_pts = torch.unsqueeze(coords, dim=0)
            else:
                patch_pts = torch.cat((patch_pts, torch.unsqueeze(coords, dim=0)), dim=0)
        
        if(all_pts_batch != None):
            all_pts_batch = torch.cat((all_pts_batch, torch.unsqueeze(patch_pts, dim=0)), dim=0)
        else:
            all_pts_batch = torch.unsqueeze(patch_pts, dim=0)

    return all_pts_batch

def resize_image(img, size=720):
    new_size = np.asarray(img.shape[:2]/np.min(img.shape[:2])*size, dtype=int)
    img = resize(img, new_size, preserve_range=True)
    img = img.astype(np.uint8)
    center = (img.shape[0]//2, img.shape[1]//2)
    
    return img[center[0]-size//2:center[0]+size//2, center[1]-size//2:center[1]+size//2, :]

def get_pixels_from_pts(img, pts, output_shape=None):
    batch_size = pts.shape[0]
    n_patches = pts.shape[1]
    n_pts = pts.shape[2]
    print('pts ', pts.shape)
    output = torch.zeros(size=(batch_size, n_patches, n_pts, 3))
    print('output ', output.shape)
    for idx_in_batch in range(batch_size):
        for idx_patch in range(n_patches):
            for idx_pt in range(n_pts):
                pt = pts[idx_in_batch, idx_patch, idx_pt]
                color = img[idx_in_batch, :, pt[1].long(), pt[0].long()]
                output[idx_in_batch, idx_patch, idx_pt] = color
    print('output ', output.shape)

    if(output_shape != None):
       output = torch.reshape(output, output_shape)
    print('output ', output.shape)

    return output


def pad_data(data, max_kpts, device):
    width = data['image_size0'][0]

    for k in data:
        if isinstance(data[k], (list, tuple)):
            new_data = []
            if(k.startswith('keypoints')):
                #padding keypoints
                for kpt in data[k]:
                    random_values = torch.randint(0, width, (max_kpts - kpt.shape[0], 2))
                    new_data += [torch.cat((kpt, random_values.to(device)), 0)]
                    
            if(k.startswith('descriptor')):
                #padding descriptors
                for desc in data[k]:
                    new_data += [F.pad(desc, 
                                (0, max_kpts - desc.shape[1]))]

            if(k.startswith('score')):
                #padding scores
                for score in data[k]:
                    new_data += [F.pad(score, 
                                (0, max_kpts - score.shape[0]))]
            data[k] = torch.stack(new_data)
    return data


# --- VISUALIZATION ---
# based on: https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/utils.py
def plot_keypoints(image, kpts, scores=None):
    kpts = np.round(kpts).astype(int)

    if scores is not None:
        # get color
        smin, smax = scores.min(), scores.max()
        assert (0 <= smin <= 1 and 0 <= smax <= 1)

        color = cm.gist_rainbow(scores * 0.4)
        color = (np.array(color[:, :3]) * 255).astype(int)[:, ::-1]
        # text = f"min score: {smin}, max score: {smax}"

        for (x, y), c in zip(kpts, color):
            c = (int(c[0]), int(c[1]), int(c[2]))
            cv2.drawMarker(image, (x, y), tuple(c), cv2.MARKER_CROSS, 6)

    else:
        for x, y in kpts:
            cv2.drawMarker(image, (x, y), (0, 0, 0), cv2.MARKER_CROSS, 6)

    return image


# based on: https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/utils.py
def plot_matches(image0, image1, kpts0, kpts1, scores=None, layout="lr"):
    """
    plot matches between two images. If score is nor None, then red: bad match, green: good match
    :param image0: reference image
    :param image1: current image
    :param kpts0: keypoints in reference image
    :param kpts1: keypoints in current image
    :param scores: matching score for each keypoint pair, range [0~1], 0: worst match, 1: best match
    :param layout: 'lr': left right; 'ud': up down
    :return:
    """
    H0, W0 = image0.shape[0], image0.shape[1]
    H1, W1 = image1.shape[0], image1.shape[1]

    if layout == "lr":
        H, W = max(H0, H1), W0 + W1
        out = 255 * np.ones((H, W, 3), np.uint8)
        out[:H0, :W0, :] = image0
        out[:H1, W0:, :] = image1
    elif layout == "ud":
        H, W = H0 + H1, max(W0, W1)
        out = 255 * np.ones((H, W, 3), np.uint8)
        out[:H0, :W0, :] = image0
        out[H0:, :W1, :] = image1
    else:
        raise ValueError("The layout must be 'lr' or 'ud'!")

    kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)

    # get color
    if scores is not None:
        smin, smax = scores.min(), scores.max()
        assert (0 <= smin <= 1 and 0 <= smax <= 1)

        color = cm.gist_rainbow(scores * 0.4)
        color = (np.array(color[:, :3]) * 255).astype(int)[:, ::-1]
    else:
        color = np.zeros((kpts0.shape[0], 3), dtype=int)
        #color[:, 1] = 255

    for (x0, y0), (x1, y1), c in zip(kpts0, kpts1, color):
        c = c.tolist()
        if layout == "lr":
            cv2.line(out, (x0, y0), (x1 + W0, y1), color=c, thickness=1, lineType=cv2.LINE_AA)
            # display line end-points as circles
            #cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
            #cv2.circle(out, (x1 + W0, y1), 2, c, -1, lineType=cv2.LINE_AA)
        elif layout == "ud":
            cv2.line(out, (x0, y0), (x1, y1 + H0), color=c, thickness=1, lineType=cv2.LINE_AA)
            # display line end-points as circles
            #cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
            #cv2.circle(out, (x1, y1 + H0), 2, c, -1, lineType=cv2.LINE_AA)

    return out
