import os
from tqdm import tqdm
import numpy as np
import glob
import cv2
import os.path as osp
import os
from pathlib import Path
import torch

from mmdet.apis import inference_detector, init_detector
# 定义工作目录
base_dir = Path('/data1/shenghao/MotionPRO/')
sub_base_dirs = []

# 遍历第一级
for first_level in base_dir.iterdir():
    if first_level.is_dir():
        # 遍历第二级
        for second_level in first_level.iterdir():
            if second_level.is_dir():
                sub_base_dirs.append(second_level)   
for working_name_dir in tqdm(sub_base_dirs):
    working_dirs = glob.glob(working_name_dir + '/*')
    working_dirs.sort()
    working_dirs = working_dirs
    print(working_dirs)

    for working_dir in tqdm(working_dirs):
        # # 创建 color 文件夹
        color_dir = os.path.join(working_dir, 'color')

        # https://github.com/open-mmlab/mmdetection/tree/master/configs/yolox
        config = 'data/mmdetection/configs/yolox/yolox_x_8x8_300e_coco.py'
        checkpoint = 'data/mmdetection/checkpoints/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth'
        # config = './mmdetection/configs/rtmdet/rtmdet_x_p6_4xb8-300e_coco.py'
        # checkpoint = './mmdetection/checkpoints/rtmdet_x_p6_4xb8-300e_coco-bf32be58.pth'
        
        # load detector
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = init_detector(config, checkpoint, device)
        
        # save results
        detection_all = []

        # mmdetection procedure
        # load all images
        img_path_list = glob.glob(osp.join(color_dir, '*.jpg'))
        img_path_list.extend(glob.glob(osp.join(color_dir, '*.png')))
        img_path_list.sort()

        print("Loading images ...")
        orig_img_bgr_all = [cv2.imread(img_path) for img_path in tqdm(img_path_list)]
        print("Image number:", len(img_path_list))
        imgs = orig_img_bgr_all

        # only take-out person (id=0)
        class_id = 0
        last_x1 = 0
        last_y1 = 0
        last_x2 = 0
        last_y2 = 0
        last_score = 0
        for i, img in enumerate(tqdm(imgs)):
            try:
                result = inference_detector(model, img)
                x1, y1, x2, y2 = result.pred_instances.bboxes[class_id].cpu()
                score = result.pred_instances.scores[class_id].cpu()
                last_x1, last_y1, last_x2, last_y2, last_score = x1, y1, x2, y2, score
            except:
                x1, y1, x2, y2, score = last_x1, last_y1, last_x2, last_y2, last_score
                with open("log.txt", "a") as file:
                    file.write(str(i) + " use last bbox \n")
            detection_all.append([i, x1, y1, x2, y2, score, 0.99, 0])
        
        # list to array
        detection_all = np.array(detection_all)
        print(detection_all.shape)
        np.save(os.path.join(working_dir, 'bbox.npy'), detection_all)