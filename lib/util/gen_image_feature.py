import cv2
import numpy as np
import glob
import os.path as osp
import os
import torch
import smplx
device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

from collections import OrderedDict
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path
from lib.model.cliff.cliff_hr48 import CLIFF as cliff_hr48

CROP_IMG_HEIGHT = 256
CROP_IMG_WIDTH = 192
CROP_ASPECT_RATIO = CROP_IMG_HEIGHT / float(CROP_IMG_WIDTH)

# Mean and standard deviation for normalizing input image
IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]

def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    # res: (height, width), (rows, cols)
    crop_aspect_ratio = res[0] / float(res[1])
    h = 200 * scale
    w = h / crop_aspect_ratio
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / w
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / w + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t

def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return np.array([round(new_pt[0]), round(new_pt[1])], dtype=int) + 1

def bbox_from_detector(bbox, rescale=1.1):
    """
    Get center and scale of bounding box from bounding box.
    The expected format is [min_x, min_y, max_x, max_y].
    """
    # center
    center_x = (bbox[0] + bbox[2]) / 2.0
    center_y = (bbox[1] + bbox[3]) / 2.0
    center = torch.tensor([center_x, center_y])

    # scale
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]
    bbox_size = max(bbox_w * CROP_ASPECT_RATIO, bbox_h)
    scale = bbox_size / 200.0
    # adjust bounding box tightness
    scale *= rescale
    return center, scale

def crop(img, center, scale, res):
    """
    Crop image according to the supplied bounding box.
    res: [rows, cols]
    """
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1)) - 1
    # Bottom right point
    br = np.array(transform([res[1] + 1, res[0] + 1], center, scale, res, invert=1)) - 1

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape, dtype=np.float32)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    try:
        new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]
    except Exception as e:
        print(e)

    new_img = cv2.resize(new_img, (res[1], res[0]))  # (cols, rows)

    return new_img, ul, br

def process_image(orig_img_rgb, bbox,
                  crop_height=256,
                  crop_width=192):
    """
    Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    """
    try:
        center, scale = bbox_from_detector(bbox)
    except Exception as e:
        print("Error occurs in person detection", e)
        # Assume that the person is centered in the image
        height = orig_img_rgb.shape[0]
        width = orig_img_rgb.shape[1]
        center = np.array([width // 2, height // 2])
        scale = max(height, width * crop_height / float(crop_width)) / 200.


    img, ul, br = crop(orig_img_rgb, center, scale, (crop_height, crop_width))
    crop_img = img.copy()

    img = img / 255.
    mean = np.array(IMG_NORM_MEAN, dtype=np.float32)
    std = np.array(IMG_NORM_STD, dtype=np.float32)
    norm_img = (img - mean) / std
    norm_img = np.transpose(norm_img, (2, 0, 1))

    return norm_img, center, scale, ul, br, crop_img

class ImageDataset(Dataset):

    def __init__(self, base_dir):
        color_dir = osp.join(base_dir, 'color')
        self.color_list = glob.glob(osp.join(color_dir, '*.png'))
        self.color_list.extend(glob.glob(osp.join(color_dir, '*.jpg')))
        self.color_list.sort()

        bbox_path = osp.join(base_dir, 'bbox.npy')
        self.bbox = np.array(np.load(bbox_path))
        print(self.bbox.shape)
        
    def __len__(self):
        return len(self.color_list)
        # return 100
    
    def __getitem__(self, idx):
        
        # color = self.color_all[idx]
        color = cv2.imread(self.color_list[idx])
        bbox = self.bbox[idx][1:5]
        img_rgb = color[:, :, ::-1]
        img_h, img_w, _ = img_rgb.shape
        focal_length = 608

        norm_img, center, scale, crop_ul, crop_br, _  = process_image(img_rgb, bbox)

        item = {}
        item["norm_img"] = norm_img
        item["center"] = center
        item["scale"] = scale
        item["crop_ul"] = crop_ul
        item["crop_br"] = crop_br
        item["img_h"] = img_h
        item["img_w"] = img_w
        item["focal_length"] = focal_length

        return item

def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not any(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict

if __name__ == "__main__":
    base_dir = Path('/data1/shenghao/MotionPRO/')
    sub_base_dirs = []

    # 遍历第一级
    for first_level in base_dir.iterdir():
        if first_level.is_dir():
            # 遍历第二级
            for second_level in first_level.iterdir():
                if second_level.is_dir():
                    sub_base_dirs.append(second_level)              

    for sub_base_dir in tqdm(sub_base_dirs):
        basedir_list = os.listdir(sub_base_dir)
        basedir_list.sort()
        print(basedir_list)

        cliff_model = torch.nn.DataParallel(cliff_hr48(smpl_mean_params='data/smpl/smpl_mean_params.npz')).to(device)
        state_dict = torch.load('data/cliff_ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt')['model']
        # state_dict = strip_prefix_if_present(state_dict, prefix="module.")
        cliff_model.load_state_dict(state_dict, strict=True)
        cliff_model.eval()

        smpl_model = smplx.create('data/smpl/SMPL_NEUTRAL.pkl').to(device)

        for basedir in tqdm(basedir_list):
            basedir = osp.join(sub_base_dir, basedir)
            print(basedir)
            output_path = osp.join(basedir, 'feature_hrnet.pth')
            if osp.exists(output_path):
                continue
            data_loader = torch.utils.data.DataLoader(ImageDataset(basedir), batch_size=1024, shuffle=False, num_workers=8)

            feature_all = []

            for i_batch, batch in tqdm(enumerate(data_loader, 0)):

                norm_img = batch["norm_img"].to(device).float()
                center = batch["center"].to(device).float()
                scale = batch["scale"].to(device).float()
                img_h = batch["img_h"].to(device).float()
                img_w = batch["img_w"].to(device).float()
                focal_length = batch["focal_length"].to(device).float()

                cx, cy, b = center[:, 0], center[:, 1], scale * 200
                bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b], dim=-1)
                # The constants below are used for normalization, and calculated from H36M data.
                # It should be fine if you use the plain Equation (5) in the paper.
                bbox_info[:, :2] = bbox_info[:, :2] / focal_length.unsqueeze(-1) * 2.8  # [-1, 1]
                bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (0.06 * focal_length)  # [-1, 1]

                with torch.no_grad():
                    feature = cliff_model(norm_img, bbox_info)   
                
                feature_all.append(feature.cpu().detach())
            
            feature_all = torch.cat(feature_all, dim=0)
            print(feature_all.shape)
            torch.save(feature_all, output_path)
            torch.cuda.empty_cache()