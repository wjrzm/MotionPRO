import numpy as np
import os
import smplx
import torch
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import os.path as osp
    
def compute_contact_region_single(point):
    point = tuple(map(int, point))

    RADIUS = 3

    x_min = point[0] - RADIUS
    y_min = point[1] - RADIUS
    x_max = point[0] + RADIUS
    y_max = point[1] + RADIUS
    
    return x_min, x_max, y_min, y_max

def compute_contact_region(point1, point2):
    """
    :param image_array: np.array, shape = (y, x) = (160, 120)
    :return: 
    """
    point1 = tuple(map(int, point1))
    point2 = tuple(map(int, point2))

    x_min = min(point1[0], point2[0]) - 5
    y_min = min(point1[1], point2[1]) - 8
    x_max = max(point1[0], point2[0]) + 5
    y_max = max(point1[1], point2[1]) + 5
    
    return x_min, x_max, y_min, y_max
    

def save_figure(f, color_image, pressure_dir):
    if not os.path.exists(pressure_dir):
        os.makedirs(pressure_dir)

    filename = f"frame_{f}_contact_region.jpg"
    save_path = os.path.join(pressure_dir, filename)
    cv2.imwrite(save_path, color_image)
    print(f"image save to: {save_path}")

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
    
    print("Pressure shape:", pressure.shape) 
    scale = (0.494/37 + 1.422/110) /2
    carpet_pos = np.zeros((pressure.shape[2], pressure.shape[1], 3))
    for i in range(pressure.shape[2]):
        for j in range(pressure.shape[1]):
            carpet_pos[i, j, 0] = -0.241 + i*scale + 0.5*scale
            carpet_pos[i, j, 1] = 1.099 -j*scale - 0.5*scale
            carpet_pos[i, j, 2] = 0.
    assert carpet_pos.shape == (37, 110, 3)
        
    
    joint_pos = smpl.joints[:, :24, :].cpu().numpy()
    assert joint_pos.shape[1] == 24 and joint_pos.shape[2] == 3

    if joint_pos.shape[0] < pressure.shape[0]:
        joint_pos = np.insert(joint_pos, 0, joint_pos[0], axis=0)
        
    contact = np.zeros((pressure.shape[0], 24))

    for f in tqdm(range(pressure.shape[0])):
        """
        joint pos and carpet pos are in the same coordinate system
        pressure array is in the different coordinate system
        """
        pressure_ = pressure[f, :, :]
        # do flip to make the image in the right direction
        pressure_ = cv2.flip(pressure_, 1)
        # pressure_ = cv2.flip(pressure_, 0)
        joint_pos_ = joint_pos[f, :, :]
        # Filter z-height
        
        for idx in range(24):
            if joint_pos[f, idx, 2] >= 0.15 :
                continue
            carpet_idx = compute_point_with_min_distance(joint_pos_[idx, :], carpet_pos)
            lxmin_a, lxmax_a, lymin_a, lymax_a = compute_contact_region_single(carpet_idx)
            carpet_pressure = pressure_[lymin_a:lymax_a, lxmin_a:lxmax_a]
            contact_sum = np.sum(carpet_pressure)
            # print(idx, contact_sum)
            if contact_sum > 0.1:
                contact[f, idx] = 1
                if save_fig:
                    img = cv2.cvtColor((pressure_*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

                    if contact_sum > 0.1:
                        cv2.circle(img, (carpet_idx[0], carpet_idx[1]), 1, (255, 0, 0), -1)

                    cv2.rectangle(img, (lxmin_a, lymin_a), (lxmax_a, lymax_a), (0, 255, 0), 1)  # 用绿色框表示左脚
                    pressure_dir = 'pressure_vis/'+str(f)+'_'+str(idx)
                    save_figure(f, img, pressure_dir)
    
    contact = contact[:, [1,2,4,5,7,8,10,11,20,21]]
    
    save_path = os.path.join(path, 'contact.npy')
    print('Contact shape {%s}, Save path {%s}'%(contact.shape, save_path))
    np.save(save_path, contact)


def compute_smpl_model(path, save_fig):
    device = torch.device('cpu')
    smpl_model_path = '/home/shenghao/code/pressure2smpl/data/smpl/SMPL_NEUTRAL.pkl'

    smpl_name = os.path.join(path, 'smpl.npy')
    pressure_name = os.path.join(path, 'pressure.npz')
    # print("Processing {} ...".format(smpl_name))
    smpl_data = np.load(smpl_name, allow_pickle=True).item()
    pressure = np.load(pressure_name)["pressure"]
    
    body_pose = torch.Tensor(smpl_data['body_pose']).reshape(-1, 23, 3).to(device)  # >>> torch.Size([17442, 23, 3])
    num_frames = body_pose.shape[0]  # >>>  17442
    betas = torch.Tensor(smpl_data['betas']).expand(num_frames, -1).to(device)  # >>> torch.Size([17442, 10])
    global_orient = torch.Tensor(smpl_data['global_orient']).unsqueeze(1).to(device)  # >>>  torch.Size([17442, 1, 3])
    transl = torch.Tensor(smpl_data['transl']).to(device)  # >>>  torch.Size([17442, 3])
    smpl_create = smplx.create(smpl_model_path).to(device)
    smpl_model = smpl_create(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl)
    
    compute_pressure_pos(pressure, smpl_model, path, save_fig)  

def gen_contact(base_dir, save_fig=False):
    for root, dirs, files in os.walk(base_dir, topdown=False):
        if 'keypoints.npy' in files and 'pressure.npz' in files and 'smpl.npy' in files:
            print("Now begin to process {%s}"%root)
            compute_smpl_model(root, save_fig)    
    
def plot_keypoints(base_dir):    
    keypoints_file = osp.join(base_dir, 'keypoints.npy')
    kps = np.load(keypoints_file)
    kps = kps.reshape(-1, 22, 3)

    smpl_left_leg = [0,1,4,7,10]
    smpl_right_leg = [0,2,5,8,11]
    smpl_left_arm = [9,13,16,18,20]
    smpl_right_arm = [9,14,17,19,21]
    smpl_head = [9,12,15]
    smpl_body = [9,6,3,0]

    CONTACT = [1,2,4,5,7,8,10,11,20,21]

    contact_file = osp.join(base_dir, 'contact.npy')
    contact = np.load(contact_file)[:,:10]

    output_dir = 'contact_vis'
    print(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ax = plt.axes(projection='3d')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    for i in tqdm(range(0, kps.shape[0])):
        contact_index = np.where(contact[i] == 1)[0]

        ax.set_xlim3d([-0.75, 0.75])
        ax.set_zlim3d([0, 2])
        ax.set_ylim3d([-0.5, 1.5])
        ax.scatter3D(kps[i,:,0], kps[i,:,1], kps[i,:,2], cmap='Greens')
        for j in list(contact_index):
            ax.scatter3D(kps[i,CONTACT[j],0], kps[i,CONTACT[j],1], kps[i,CONTACT[j],2], color='Red', s=80)
        
        ax.plot3D(kps[i,smpl_left_leg,0], kps[i,smpl_left_leg,1], kps[i,smpl_left_leg,2], 'red')
        ax.plot3D(kps[i,smpl_left_arm,0], kps[i,smpl_left_arm,1], kps[i,smpl_left_arm,2], 'red')
        ax.plot3D(kps[i,smpl_right_leg,0], kps[i,smpl_right_leg,1], kps[i,smpl_right_leg,2], 'red')
        ax.plot3D(kps[i,smpl_right_arm,0], kps[i,smpl_right_arm,1], kps[i,smpl_right_arm,2], 'red')
        ax.plot3D(kps[i,smpl_head,0], kps[i,smpl_head,1], kps[i,smpl_head,2], 'red')
        ax.plot3D(kps[i,smpl_body,0], kps[i,smpl_body,1], kps[i,smpl_body,2], 'red')
        # plt.show()
        plt.savefig(output_dir +"/" +str(i).zfill(5)+".jpg")

        ax.clear()

if __name__ == '__main__':
    base_dir = '/data1/shenghao/MotionPRO'
    gen_contact(base_dir, False)
    plot_keypoints(base_dir)
    



    

        