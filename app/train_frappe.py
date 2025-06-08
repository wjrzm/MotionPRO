import torch
import torch.nn as nn
import hydra
import os
import os.path as osp
import numpy as np
import smplx
import copy

from tqdm import tqdm
from loguru import logger as log
from torch.utils.tensorboard import SummaryWriter

from lib.model.FRAPPE import FRAPPE
from lib.dataset.image_pressure import ImagePressureDataset

class Loss(nn.Module):
    def __init__(self, weight):
        super(Loss, self).__init__()
        self.weight = weight
        self.mse = torch.nn.MSELoss()
        self.mae_smooth = torch.nn.SmoothL1Loss()

        self.contactIds = [1,2,4,5,7,8,10,11,20,21]

    def cal_footloss(self, pred, target, contact):

        loss_contact = torch.sum((pred - target) ** 2, dim=2)
        contact_num = torch.sum(contact)
        loss = torch.sum(loss_contact * contact) / contact_num

        return loss

    def forward(self, pred, target, contact):
        loss_theta = self.mse(pred['theta'], target['theta'])
        loss_trans = self.mae_smooth(pred['trans'], target['trans'])
        loss_2d_joint = self.mae_smooth(pred['joint'][:,:,[0,2]]-pred['joint'][:,0:1,[0,2]], target['joint'][:,:,[0,2]]-target['joint'][:,0:1,[0,2]])
        loss_joint = self.mae_smooth(pred['joint']-pred['joint'][:,[0]], target['joint']-target['joint'][:,[0]])
        loss_foot = self.cal_footloss(pred['joint'][:, self.contactIds], target['joint'][:, self.contactIds], contact)
        
        loss = loss_theta*self.weight['theta']+ loss_trans*self.weight['trans']+ loss_2d_joint*self.weight['2d_joint']+ loss_joint*self.weight['joint'] + loss_foot*self.weight['foot']

        loss_item = {
            'loss': loss,
            'loss_theta': loss_theta,
            'loss_trans': loss_trans,
            'loss_2d_joint': loss_2d_joint,
            'loss_joint': loss_joint,
            'loss_foot': loss_foot,
        }

        return loss_item
    

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg['task']['gpu'])

    # manual log
    hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    hydra_name = hydra.core.hydra_config.HydraConfig.get().job.name
    log.add(os.path.join(hydra_path, hydra_name+'.log'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    log.info(f"Working directory: {os.getcwd()}")
    log.info(f"Configuration: {cfg}")
    log.info(f"PyTorch version: {torch.__version__}")

    task_info = os.path.join(str(cfg['task']['name']), str(cfg['task']['loss_name']), str(cfg['task']['learning_rate']))

    tensorboard_path = os.path.join('data/tensorboard', task_info)
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    writer = SummaryWriter(tensorboard_path)

    checkpoint_dir = os.path.join(cfg['task']['checkpoint_dir'], task_info)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    result_dir = os.path.join(cfg['task']['result_dir'], task_info)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    np.random.seed(0)
    torch.manual_seed(0)
    model = torch.nn.DataParallel(FRAPPE()) # model
    smpl = smplx.create('data/smpl/SMPL_NEUTRAL.pkl').to(device)

    train_dataset = ImagePressureDataset(cfg, 'train')
    eval_dataset = ImagePressureDataset(cfg, 'eval')

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['task']['batch_size'], shuffle=False, pin_memory=True, num_workers=4)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=cfg['task']['batch_size'], shuffle=False, pin_memory=True, num_workers=4)

    model.to(device=device, non_blocking=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['task']['learning_rate'], weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=cfg['task']['scheduler_patience'], verbose=True)
    
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info (f'pytorch_total_params: {pytorch_total_params}')

    start_epoch = 0
    best_eval_loss = np.inf

    weight = {
        'theta': cfg['task']['lamda_theta'],
        'trans': cfg['task']['lamda_trans'],
        '2d_joint': cfg['task']['lamda_2d_joint'],
        'joint': cfg['task']['lamda_joint'],
        'foot': cfg['task']['lamda_foot'],
    }
    
    loss = Loss(weight)

    if cfg['task']['train_continue']:

        checkpoint = torch.load(cfg['task']['checkpoint_path'])
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        log.info("Now continue training")
        log.info(f"Start epoch: {start_epoch}")

    for epoch in tqdm(range(start_epoch, cfg['task']['epochs'])):

        smpl_params_all = []
        joint_all = []
        
        loss_train = {
            'loss': 0.0,
            'loss_theta': 0.0,
            'loss_trans': 0.0,
            'loss_2d_joint': 0.0,
            'loss_joint': 0.0,
            'loss_foot': 0.0
        }

        loss_eval = copy.deepcopy(loss_train)
        
        for i_batch, item in enumerate(train_dataloader, 0):
            model.train(True)
            optimizer.zero_grad()

            for key in item:
                if key != 'pressure' and key != 'feature':
                    if key == 'joint':
                        item[key] = item[key].reshape(-1,22,3)
                    else:
                        item[key] = item[key].reshape(-1, item[key].shape[-1])
                item[key] = item[key].to(device=device, non_blocking=True)
            
            with torch.set_grad_enabled(True):
                smpl_params = model(item['feature'], item['pressure'])
                pred_beta, pred_theta, pred_trans = torch.split(smpl_params, [10, 72, 3], dim=1)
                pred_global_orient, pred_body_pose = torch.split(pred_theta, [3, 69], dim=1)
                smpl_result = smpl(betas=item['beta'], # shape parameters
                    body_pose=pred_body_pose, # pose parameters
                    global_orient=pred_global_orient, # global orientation
                    transl=pred_trans) # global translation
                pred_joint = smpl_result.joints[:, :22]

            pred_train = {
                'beta': pred_beta,
                'theta': pred_theta,
                'trans': pred_trans,
                'joint': pred_joint
            }

            target_train = {
                'beta': item['beta'],
                'theta': item['theta'],
                'trans': item['trans'],
                'joint': item['joint']
            }
            
            loss_item_train = loss(pred_train, target_train, item['contact'])

            loss_item_train['loss'].backward()
            optimizer.step()

            for key in loss_item_train:
                loss_train[key] += loss_item_train[key].item()
            
        log.info("Now running on val set")

        for i_batch, item in enumerate(eval_dataloader, 0):
            model.eval()
            for key in item:
                if key != 'pressure' and key != 'feature':
                    if key == 'joint':
                        item[key] = item[key].reshape(-1,22,3)
                    else:
                        item[key] = item[key].reshape(-1, item[key].shape[-1])
                item[key] = item[key].to(device=device, non_blocking=True)

            with torch.no_grad():
                smpl_params = model(item['feature'], item['pressure'])
                smpl_params_all.append(smpl_params.cpu().detach().numpy())
                pred_beta, pred_theta, pred_trans = torch.split(smpl_params, [10, 72, 3], dim=1)
                pred_global_orient, pred_body_pose = torch.split(pred_theta, [3, 69], dim=1)

                smpl_result = smpl(betas=item['beta'], # shape parameters
                    body_pose=pred_body_pose, # pose parameters
                    global_orient=pred_global_orient, # global orientation
                    transl=pred_trans) # global translation
                pred_joint = smpl_result.joints[:, :22]
                joint_all.append(pred_joint.cpu().detach().numpy())

            pred_eval = {
                'beta': pred_beta,
                'theta': pred_theta,
                'trans': pred_trans,
                'joint': pred_joint
            }

            target_eval = {
                'beta': item['beta'],
                'theta': item['theta'],
                'trans': item['trans'],
                'joint': item['joint']
            }

            loss_item_eval = loss(pred_eval, target_eval, item['contact'])

            for key in loss_item_eval:
                loss_eval[key] += loss_item_eval[key].item()

        for key in loss_train:
            writer.add_scalar(f'train/{key}', loss_train[key], epoch)
        for key in loss_eval:
            writer.add_scalar(f'eval/{key}', loss_eval[key], epoch)

        scheduler.step(loss_eval['loss'])

        last_checkpoint_path = osp.join(checkpoint_dir, f'imagepressure2smpl_{epoch-1}.pth')
        if osp.exists(last_checkpoint_path):
            os.remove(last_checkpoint_path)
        
        if loss_eval['loss'] < best_eval_loss:
            best_eval_loss = loss_eval['loss']
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'eval_loss': loss_eval},
            osp.join(checkpoint_dir, f'imagepressure2smpl_best.pth'))

            eval_smpl_params = np.concatenate(np.array(smpl_params_all), axis=0)
            joint_all = np.concatenate(np.array(joint_all), axis=0)
            output = {'smpl_params': eval_smpl_params,
                      'joint': joint_all}
            output_file = os.path.join(result_dir, f'eval_result_best.npy')
            np.save(output_file, output)
        else:
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'eval_loss': loss_eval},
            osp.join(checkpoint_dir, f'imagepressure2smpl_{epoch}.pth'))
        
            eval_smpl_params = np.concatenate(np.array(smpl_params_all), axis=0)
            joint_all = np.concatenate(np.array(joint_all), axis=0)
            output = {'smpl_params': eval_smpl_params,
                        'joint': joint_all}
            output_file = os.path.join(result_dir, f'eval_result_latest.npy')
            np.save(output_file, output)

        log.info(f"Epoch: {epoch}")
        log.info("Train Loss: %.6f, Evaluate Loss: %.6f" % (loss_train['loss'], loss_eval['loss']))

if __name__ == '__main__':
    main()