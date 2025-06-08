import torch
import numpy as np
import smplx
from lib.util.io import findAllFilesWithSpecifiedName

def generate_kps(gt_files):
    device = torch.device('cpu')
    dtype = torch.float32
    smpl = smplx.create('data/smpl/SMPL_NEUTRAL.pkl').to(device)
    for gt_file in gt_files:
        smpl_gt = np.load(gt_file, allow_pickle=True).item()
        betas = torch.tensor(smpl_gt['betas'][:10], dtype=dtype, device=device).unsqueeze(0)
        body_pose = torch.tensor(smpl_gt['body_pose'], dtype=dtype, device=device).reshape(-1, 23, 3)
        global_orient = torch.tensor(smpl_gt['global_orient'], dtype=dtype,device=device).unsqueeze(1)
        transl = torch.tensor(smpl_gt['transl'], dtype=dtype, device=device)
        # print(betas.shape, body_pose.shape, global_orient.shape, transl.shape)
        result = smpl(betas=betas, # shape parameters
                    body_pose=body_pose, # pose parameters
                    global_orient=global_orient, # global orientation
                    transl=transl) # global translation
        
        keypoints = result.joints[:, :24].cpu().numpy()
        print(keypoints.shape)
        output_file_kps = gt_file.replace('smpl.npy', 'keypoints.npy')
        print(output_file_kps)
        np.save(output_file_kps, keypoints)

if __name__ == '__main__':
    gt_files = findAllFilesWithSpecifiedName('/data1/shenghao/MotionPRO', 'smpl.npy')
    gt_files.sort()

    for gt_file in gt_files:
        print(gt_file)
        generate_kps([gt_file])