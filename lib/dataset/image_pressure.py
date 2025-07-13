import torch
import numpy as np

from torch.utils.data import Dataset

from lib.util.io import findAllFilesWithSpecifiedName

class ImagePressureDataset(Dataset):

    def __init__(self, cfg, mode):
        self.mode = mode
        self.window_length = cfg['task']['window_length']
        self.dtype = torch.float32

        if self.mode == 'train':
            self.feature_files = findAllFilesWithSpecifiedName(cfg['task']['train_data_dir'], 'feature_hrnet.pth')
            print(self.feature_files)

        elif self.mode == 'eval':
            self.feature_files = findAllFilesWithSpecifiedName(cfg['task']['eval_data_dir'], 'feature_hrnet.pth')
            print(self.feature_files)

        self.gt_kps = []
        self.gt_contact = []
        self.gt_verts = []
        self.gt_smpl = []
        self.feature_all = []
        self.pressure_all = []
        self.index_list = [0]
        self.index_origin_list = [0]

        for idx, feature_file in enumerate(self.feature_files):

            feature = torch.load(feature_file)
            self.feature_all.append(feature)
            
            pressure_file = feature_file.replace('feature_hrnet.pth', 'pressure.npz')
            gt_kps_file = feature_file.replace('feature_hrnet.pth', 'keypoints.npy')
            gt_smpl_file = feature_file.replace('feature_hrnet.pth', 'smpl.npy')
            contact_file = feature_file.replace('feature_hrnet.pth', 'contact.npy')

            pressure = np.load(pressure_file)['pressure']
            kps_gt = np.load(gt_kps_file)
            contact_gt = np.load(contact_file)
            smpl_gt = np.load(gt_smpl_file, allow_pickle=True).item()

            self.pressure_all.append(pressure)
            self.gt_kps.append(kps_gt)
            self.gt_smpl.append(smpl_gt)
            self.gt_contact.append(contact_gt)

            self.index_origin_list.append(kps_gt.shape[0]+self.index_origin_list[-1])

            if kps_gt.shape[0] % self.window_length == 0:
                self.index_list.append(kps_gt.shape[0]//self.window_length+self.index_list[-1])
            elif kps_gt.shape[0] % self.window_length != 0:
                self.index_list.append((kps_gt.shape[0]//self.window_length)+1+self.index_list[-1])

        print(self.index_origin_list)
        print(self.index_list)
        print('Lens: ', self.index_list[-1])
        
    def __len__(self):
        return self.index_list[-1]
    
    def __getitem__(self, idx):
        item = {}

        for i in range(len(self.index_list)-1):
            if idx >= self.index_list[i] and idx < self.index_list[i+1]:
                idx_in_file = idx - self.index_list[i]
                origin_max_index = self.index_origin_list[i+1] - self.index_origin_list[i]
                file_index = i
                break
                
        window_left = idx_in_file*self.window_length
        window_right = min(origin_max_index, (idx_in_file+1)*self.window_length)
        pressure = self.pressure_all[file_index][window_left:window_right, :] / 255
        feature = self.feature_all[file_index][window_left:window_right, :]
        keypoint = self.gt_kps[file_index][window_left:window_right,:22]
        contact = self.gt_contact[file_index][window_left:window_right,:10]
        smpl_gt = self.gt_smpl[file_index]

        beta = torch.from_numpy(smpl_gt['betas'][:10]).repeat(self.window_length, 1).float()
        body_pose = torch.from_numpy(smpl_gt['body_pose'][window_left:window_right]).float()
        global_orient = torch.from_numpy(smpl_gt['global_orient'][window_left:window_right]).float()
        transl = torch.from_numpy(smpl_gt['transl'][window_left:window_right]).float()
        theta = torch.cat([global_orient, body_pose], dim=1)

        pressure = torch.from_numpy(pressure).float()
        keypoint = torch.from_numpy(keypoint).float()
        contact = torch.from_numpy(contact).float()

        if feature.shape[0] < self.window_length:
            pressure = torch.cat([pressure, torch.zeros(self.window_length-pressure.shape[0], pressure.shape[1], pressure.shape[2], dtype=self.dtype)], dim=0)
            feature = torch.cat([feature, torch.zeros(self.window_length-feature.shape[0], feature.shape[1], dtype=self.dtype)], dim=0)
            keypoint = torch.cat([keypoint, torch.zeros(self.window_length-keypoint.shape[0], keypoint.shape[1], keypoint.shape[2], dtype=self.dtype)], dim=0)
            contact = torch.cat([contact, torch.zeros(self.window_length-contact.shape[0],contact.shape[1], dtype=self.dtype)], dim=0)
            beta = torch.cat([beta, torch.zeros(self.window_length-beta.shape[0], beta.shape[1], dtype=self.dtype)], dim=0)
            theta = torch.cat([theta, torch.zeros(self.window_length-theta.shape[0], theta.shape[1], dtype=self.dtype)], dim=0)
            transl = torch.cat([transl, torch.zeros(self.window_length-transl.shape[0], transl.shape[1], dtype=self.dtype)], dim=0)

        item['pressure'] = pressure
        item['feature'] = feature
        
        item['beta'] = beta
        item['theta'] = theta
        item['trans'] = transl
        item['joint'] = keypoint
        item['contact'] = contact

        return item
    
