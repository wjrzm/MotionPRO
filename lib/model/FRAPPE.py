import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from lib.util.geometry import rotation_matrix_to_angle_axis, rot6d_to_rotmat

class TemporalEncoder(nn.Module):
    def __init__(
            self,
            n_layers=1,
            input_size=512,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True
    ):
        super(TemporalEncoder, self).__init__()

        self.input_size = input_size
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=n_layers
        )

        self.linear = None
        if bidirectional:
            self.linear = nn.Linear(hidden_size * 2, input_size)
        elif add_linear:
            self.linear = nn.Linear(hidden_size, input_size)
        self.use_residual = use_residual

    def forward(self, x):
        n, t, f = x.shape
        x = x.permute(1, 0, 2)  # NTF -> TNF
        y, _ = self.gru(x)
        if self.linear:
            y = F.relu(y)
            y = self.linear(y.view(-1, y.size(-1)))
            y = y.view(t, n, f)
        if self.use_residual and y.shape[-1] == self.input_size:
            y = y + x
        y = y.permute(1, 0, 2)  # TNF -> NTF
        return y

class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        
        npose = 24 * 6

        self.fc1 = nn.Linear(512 * 4 + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.dectrans = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)

        mean_params = np.load('data/smpl/smpl_mean_params.npz')
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        init_trans = torch.zeros_like(init_cam)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_trans', init_trans)



    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, init_trans=None, n_iter=3, J_regressor=None):
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_trans is None:
            init_trans = self.init_trans.expand(batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_trans = init_trans

        for i in range(n_iter):
            xc = torch.cat([x, pred_pose, pred_shape, pred_trans], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_trans = self.dectrans(xc) + pred_trans

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)
        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)

        output = torch.cat([pred_shape, pose, pred_trans], 1)
        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in = 1024, d_hid = 1024, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.gelu(self.w_1(x))) # Feed Forward
        
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x) # add & norm
        
        return x

class FRAPPE(nn.Module):
    def __init__(self):
        super(FRAPPE, self).__init__()   #tactile 96*96

        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=0, bias=False)
        self.resnet.maxpool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2048)

        self.cross_attention = nn.MultiheadAttention(2048, 4)
        self.self_attention = nn.MultiheadAttention(2048, 4)
        self.ffn = PositionwiseFeedForward(d_in=2048, d_hid=2048, dropout=0.1)

        self.gru = TemporalEncoder(
            n_layers=2,
            input_size=2048,
            hidden_size=2048,
            bidirectional=True,
            add_linear=False,
            use_residual=True,
        )

        self.regressor = Regressor()

        self.fc_output = nn.Linear(4096, 2048)

    def forward(self, img_feature, pressure):
        res_img = img_feature
        img_feature = self.gru(img_feature)

        batchsize, windowSize, height, width = pressure.size()
        pressure = pressure.view(batchsize*windowSize, 1, height, width)
        pressure_feature = self.resnet(pressure)
        pressure_feature = pressure_feature.view(batchsize, windowSize, -1)
        
        res_pressure = pressure_feature
        pressure_feature = self.gru(pressure_feature)
        pressure_feature = pressure_feature + res_pressure
        res_pressure = pressure_feature
        pressure_feature = self.gru(pressure_feature)
        
        pressure_feature = pressure_feature + res_pressure
        img_feature = img_feature + res_img

        res_img = img_feature
        res_pressure = pressure_feature

        img_feature,_ = self.self_attention(img_feature, img_feature, img_feature)
        pressure_feature,_ = self.self_attention(pressure_feature, pressure_feature, pressure_feature)
        
        img_feature = img_feature + res_img
        pressure_feature = pressure_feature + res_pressure

        res_img = img_feature
        res_pressure = pressure_feature
        res_pressure_sa = pressure_feature

        fusion_feature,_ = self.cross_attention(pressure_feature, img_feature, img_feature)
        fusion_feature = fusion_feature + res_img + res_pressure
        fusion_feature = self.ffn(fusion_feature)

        fusion_feature = torch.cat([fusion_feature, res_pressure_sa], dim=2)
        fusion_feature = self.fc_output(fusion_feature)
        fusion_feature = fusion_feature.reshape(-1, fusion_feature.size(-1))
        output = self.regressor(fusion_feature)

        return output