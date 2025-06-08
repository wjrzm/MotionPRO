import numpy as np
import os
import smplx
import torch
import matplotlib
matplotlib.use('TkAgg')  # 使用 TkAgg 后端, 避免和cv2冲突
import cv2
from lib.util.io import findAllFilesWithSpecifiedName

def compute_contact_region_single(point):
    # 确保 point1 和 point2 是整数元组
    point = tuple(map(int, point))

    RADIUS = 3

    x_min = point[0] - RADIUS
    y_min = point[1] - RADIUS
    x_max = point[0] + RADIUS
    y_max = point[1] + RADIUS

    return x_min, x_max, y_min, y_max

def save_figure(f, color_image, pressure_dir):
    if not os.path.exists(pressure_dir):
        os.makedirs(pressure_dir)

    # 保存带有 bounding box 的图片
    filename = f"frame_{f}_contact_region.jpg"
    save_path = os.path.join(pressure_dir, filename)
    cv2.imwrite(save_path, color_image)
    print(f"图片已保存到: {save_path}")

def compute_point_with_min_distance(point, points):
    """
    :param point: np.array, shape = (3,)
    :param points: np.array, shape = (x, y, 3)
    :return: 
    """
    distances = np.linalg.norm(points - point, axis=2)
    min_idx = np.unravel_index(np.argmin(distances), distances.shape)
    return min_idx

def compute_pressure_pos(pressure, smpl, path, save_fig=False):
    """
    :param pressure: np.array, shape = (num_frame, y, x) = (num_frame, 160, 120)
    :param smpl: smplx model
    """
    pressure_dir = path +'/pressure'
    print("Pressure shape:", pressure.shape)  # >>> Pressure shape: (17604, 160, 120)
    scale = 2.0/pressure.shape[1]
    carpet_pos = np.zeros((pressure.shape[2], pressure.shape[1], 3))
    for i in range(pressure.shape[2]):
        for j in range(pressure.shape[1]):
            carpet_pos[i, j, 0] = i*scale + 0.5*scale
            carpet_pos[i, j, 1] = -j*scale - 0.5*scale
            carpet_pos[i, j, 2] = 0.
    print("Carpet pos shape:", carpet_pos.shape)  # >>> Carpet pos shape: (120, 160, 3)
    
    lankle = 7
    rankle = 8
    lfoot = 10
    rfoot = 11  # lankle = SMPL_joint_set.joint_names.index("left_ankle")
    joint_pos = smpl.joints[:, :24, :].cpu().numpy()
    print("Joint pos shape:", joint_pos.shape)
    
    num_contact = [len(np.nonzero(pressure[f, :, :])[0]) for f in range(pressure.shape[0])]
    max_contact = max(num_contact)
    print("Max contact:", max_contact)
    
    pressure_processed = []
    contact_sum = []
    pressure_single = []
    
    contact = np.zeros((pressure.shape[0], 4))

    for f in range(pressure.shape[0]):
        """
        joint pos and carpet pos are in the same coordinate system
        pressure array is in the different coordinate system
        """
        pressure_ = pressure[f, :, :]
        
        # do flip to make the image in the right direction
        pressure_ = cv2.flip(pressure_, 1)
        pressure_ = cv2.flip(pressure_, 0)
        
        joint_pos_ = joint_pos[f, :, :]
        
        la_contact_idx = compute_point_with_min_distance(joint_pos_[lankle, :], carpet_pos)
        ra_contact_idx = compute_point_with_min_distance(joint_pos_[rankle, :], carpet_pos)
        lf_contact_idx = compute_point_with_min_distance(joint_pos_[lfoot, :], carpet_pos)
        rf_contact_idx = compute_point_with_min_distance(joint_pos_[rfoot, :], carpet_pos)
        
        lxmin_a, lxmax_a, lymin_a, lymax_a = compute_contact_region_single(la_contact_idx)
        lxmin_f, lxmax_f, lymin_f, lymax_f = compute_contact_region_single(lf_contact_idx)
        rxmin_a, rxmax_a, rymin_a, rymax_a = compute_contact_region_single(ra_contact_idx)
        rxmin_f, rxmax_f, rymin_f, rymax_f = compute_contact_region_single(rf_contact_idx)
        
        # 计算该区域内的像素之和
        lapressure = pressure_[lymin_a:lymax_a, lxmin_a:lxmax_a]
        lfpressure = pressure_[lymin_f:lymax_f, lxmin_f:lxmax_f]
        rapressure = pressure_[rymin_a:rymax_a, rxmin_a:rxmax_a]
        rfpressure = pressure_[rymin_f:rymax_f, rxmin_f:rxmax_f]

        lacontact_sum = np.sum(lapressure)
        lfcontact_sum = np.sum(lfpressure)
        racontact_sum = np.sum(rapressure)
        rfcontact_sum = np.sum(rfpressure)
        # print("LA Contact Sum:", lcontact_sum, ", RA Contact Sum:", rcontact_sum)
        
        if lacontact_sum > 100:
            contact[f, 0] = 1
        if lfcontact_sum > 100:
            contact[f, 1] = 1
        if racontact_sum > 100:
            contact[f, 2] = 1
        if rfcontact_sum > 100:
            contact[f, 3] = 1

        if save_fig:
            # 创建一个彩色图像用于绘制左右脚的 bounding box
            img = cv2.cvtColor(pressure_, cv2.COLOR_GRAY2BGR)
            # img = cv2.normalize(img, None, 100, 255, cv2.NORM_MINMAX)  # 调整亮度，使背景更亮
            
            # 标记左右脚的接触点
            # img[la_contact_idx[1], la_contact_idx[0]] = [0, 0, 255]
            # img[ra_contact_idx[1], ra_contact_idx[0]] = [0, 0, 255]
            # img[lf_contact_idx[1], lf_contact_idx[0]] = [0, 0, 255]
            # img[rf_contact_idx[1], rf_contact_idx[0]] = [0, 0, 255]
            
            if lacontact_sum > 100:
                cv2.circle(img, (la_contact_idx[0], la_contact_idx[1]), 1, (255, 0, 0), -1)
            if lfcontact_sum > 100:
                cv2.circle(img, (lf_contact_idx[0], lf_contact_idx[1]), 1, (255, 0, 0), -1)
            if racontact_sum > 100:
                cv2.circle(img, (ra_contact_idx[0], ra_contact_idx[1]), 1, (255, 0, 0), -1)
            if rfcontact_sum > 100:
                cv2.circle(img, (rf_contact_idx[0], rf_contact_idx[1]), 1, (255, 0, 0), -1)

            cv2.rectangle(img, (lxmin_a, lymin_a), (lxmax_a, lymax_a), (0, 255, 0), 1)  # 用绿色框表示左脚
            cv2.rectangle(img, (lxmin_f, lymin_f), (lxmax_f, lymax_f), (0, 255, 0), 1)  # 用绿色框表示左脚

            cv2.rectangle(img, (rxmin_a, rymin_a), (rxmax_a, rymax_a), (0, 0, 255), 1)  # 用红色框表示右脚
            cv2.rectangle(img, (rxmin_f, rymin_f), (rxmax_f, rymax_f), (0, 0, 255), 1)  # 用红色框表示右脚
            print("LA Pressure:", img[la_contact_idx[1], la_contact_idx[0]], ", LF Pressure:", img[lf_contact_idx[1], lf_contact_idx[0]], ", RA Pressure:", img[ra_contact_idx[1], ra_contact_idx[0]], ", RF Pressure:", img[rf_contact_idx[1], rf_contact_idx[0]])
            
            # 用白色矩形框在原图上标记绘制bounding box
            # cv2.rectangle(pressure_, (lxmin, lymin), (lxmax, lymax), (255, 255, 255), 2)  # 白色矩形框
            # cv2.rectangle(pressure_, (rxmin, rymin), (rxmax, rymax), (255, 255, 255), 2)  # 白色矩形框
        
            save_figure(f, img, pressure_dir)
        
    pressure_processed = np.array(pressure_processed)
    print(path)
    np.save(path + "/contact.npy", contact)

    return pressure_processed, contact_sum, pressure_single


def compute_smpl_model(path):
    device = torch.device('cpu')
    smpl_model_path = 'data/smpl/SMPL_NEUTRAL.pkl'

    smpl_name = os.path.join(path, 'smpl.npy')
    pressure_name = os.path.join(path, 'pressure.npz')
    print("Processing {} ...".format(smpl_name))
    smpl_data = np.load(smpl_name, allow_pickle=True).item()
    pressure = np.load(pressure_name)["pressure"]
    
    body_pose = torch.Tensor(smpl_data['body_pose']).reshape(-1, 23, 3).to(device)  # >>> torch.Size([17442, 23, 3])
    num_frames = body_pose.shape[0]  # >>>  17442
    betas = torch.Tensor(smpl_data['betas']).expand(num_frames, -1).to(device)  # >>> torch.Size([17442, 10])
    global_orient = torch.Tensor(smpl_data['global_orient']).unsqueeze(1).to(device)  # >>>  torch.Size([17442, 1, 3])
    transl = torch.Tensor(smpl_data['transl']).to(device)  # >>>  torch.Size([17442, 3])
    smpl_create = smplx.create(model_path=smpl_model_path).to(device)
    smpl_model = smpl_create(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl)
    
    pressure_processed, contact_sum, pressure_single = compute_pressure_pos(pressure, smpl_model, path)  # pressure in bounding box:  list: num_frames * 2 * np.array([y, x])

    return pressure_processed, contact_sum, pressure_single

if __name__ == '__main__':
    base_dir = '/data1/shenghao/MotionPRO/'
    file_list = findAllFilesWithSpecifiedName(base_dir, 'pressure.npz')
    # print(file_list)
    for file_path in file_list:
        base_dir = os.path.dirname(file_path)
        print("Processing {} ...".format(base_dir))
        pressure_processed, contact_sum, pressure_single = compute_smpl_model(base_dir)