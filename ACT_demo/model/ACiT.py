import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
from model.dnns import Tokenizer, SinCosPosEncoding
from sklearn.model_selection import StratifiedShuffleSplit


class RegLabelNorm(nn.Module):
    def __init__(self):
        super(RegLabelNorm, self).__init__()
        self.mean = 0
        self.scale = 1
        self.is_fit = False
        self.weight = nn.Parameter(torch.Tensor([1]))
        self.bias = nn.Parameter(torch.Tensor([0]))

    def fit(self, data):
        self.mean = np.mean(data)
        self.scale = (np.max(data) - np.min(data))
        self.is_fit = True

    def norm(self, y):
        assert self.is_fit is True, 'Label Normalizer not Fitted yet'
        return ((y - self.mean) / self.scale) * self.weight + self.bias

    def de_norm(self, y):
        assert self.is_fit is True, 'Label Normalizer not Fitted yet'
        re_y = (y - self.bias) * self.scale / self.weight + self.mean
        return re_y.reshape(-1, 1)


class ACiNet(nn.Module):
    def __init__(self, configs):
        super(ACiNet, self).__init__()
        self.token = Tokenizer(configs.aci_token, stride=configs.aci_stride)
        self.pe = SinCosPosEncoding()
        dp_rate = configs.aci_dp

        token_len = configs.aci_token * 2 + 1
        self.token_l = token_len
        band_num = int(np.ceil(configs.band_num / configs.aci_stride))
        self.stride = configs.aci_stride

        self.encoder1 = SpectrA(d_model=token_len, nhead=configs.aci_nhead, dropout=dp_rate,
                                dim_feedforward=configs.aci_ff, band_num=band_num)
        self.fc1 = nn.Linear(band_num, configs.aci_fc1)
        self.dp_out1 = nn.Dropout(p=dp_rate)
        self.fc2 = nn.Linear(configs.aci_fc1, configs.aci_fc2)
        self.dp_out2 = nn.Dropout(p=dp_rate)
        self.fc3 = nn.Linear(configs.aci_fc2, configs.y_dim)
        self.bn1 = nn.BatchNorm1d(configs.aci_fc1)
        self.bn2 = nn.BatchNorm1d(configs.aci_fc2)
        self.configs = configs

        self.alpha1 = nn.Parameter(torch.Tensor([0.8]))
        self.alpha2 = nn.Parameter(torch.Tensor([0.2]))
        self.beta1 = nn.Parameter(torch.ones([token_len, token_len]))
        self.token_v = nn.Linear(token_len, token_len)

        self.x_mean = None
        self.x_scale = None

    def load_corr_map(self, corr_map):
        corr_map = corr_map[::self.stride, ::self.stride]
        self.encoder1.get_corr_map(corr_map)

    def load_data_scale(self, x, whole=True):
        self.x_mean = torch.min(x[:, 20:-20])
        self.x_scale = torch.abs(torch.max(x[:, 20:-20] - self.x_mean))

    def forward(self, x, basel, cali_spec=None):
        x = (x - self.x_mean) / self.x_scale
        basel = basel[:, ::self.stride]

        tx = self.token.transform(x)

        pe = self.pe.get_pe(tx)
        tx = tx + pe
        if cali_spec is not None:
            cali_spec = (cali_spec - self.x_mean) / self.x_scale

            tx1 = self.token.transform(cali_spec)

            pe1 = self.pe.get_pe(tx1)
            cali_spec = tx1 + pe1
            att_f = self.encoder1(tx, cali_spec)
        else:
            att_f = self.encoder1(tx)
        att_f = torch.mul(self.token_v(att_f), F.softmax(torch.matmul(att_f, self.beta1), dim=-1))
        att_f = torch.sum(att_f.squeeze(), dim=-1)

        fea = self.alpha1 * att_f + self.alpha2 * basel

        fc1 = self.dp_out1(F.relu(self.fc1(fea)))

        fc1 = self.bn1(fc1)

        fc2 = self.dp_out2(F.relu(self.fc2(fc1)))

        res = self.fc3(self.bn2(fc2))

        if self.configs.task == 'classification':
            res = F.softmax(res, dim=1)
        else:
            res = F.tanh(res)

        return res


class SpectrA(nn.Module):
    def __init__(self, d_model, nhead, dropout, dim_feedforward, band_num):
        super(SpectrA, self).__init__()
        self.multi_head_att = _multihead_spectral_att(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, band_num=band_num)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.ff_1 = nn.Linear(d_model, dim_feedforward)
        self.bn2 = nn.BatchNorm1d(dim_feedforward)
        # self.bn2 = nn.BatchNorm1d(64)
        self.ff_2 = nn.Linear(dim_feedforward, d_model)
        self.bn3 = nn.BatchNorm1d(d_model)

        self.ff_dp = nn.Dropout(p=dropout)
        self.corr_map = None

    def get_corr_map(self, corr_map):
        self.corr_map = corr_map

    def forward(self, x, cali_spec=None):
        if self.corr_map is None:
            self.corr_map = 0
            warnings.warn('empty correlation map')
        if cali_spec is None:
            ma_res, ma_weight = self.multi_head_att(x, self.corr_map)
        else:
            ma_res, ma_weight = self.multi_head_att(x, self.corr_map, cali_spec)
        ma_res = ma_res + x
        ma_res = self.bn1(ma_res.permute(0, 2, 1)).permute(0, 2, 1)

        ff_res = F.relu(self.ff_1(ma_res))

        ff_res = self.bn2(ff_res.permute(0, 2, 1)).permute(0, 2, 1)

        ff_res = F.relu(self.ff_2(ff_res))

        ff_res = self.ff_dp(ff_res)
        ff_res = ma_res + ff_res
        res = self.bn3(ff_res.permute(0, 2, 1)).permute(0, 2, 1)
        return res


class _multihead_spectral_att(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, band_num):
        super(_multihead_spectral_att, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim and num_head error"
        d_kqv = embed_dim // num_heads
        self.Wq = nn.Parameter(torch.zeros(num_heads, embed_dim, d_kqv))
        self.Wk = nn.Parameter(torch.zeros(num_heads, embed_dim, d_kqv))
        self.Wv = nn.Parameter(torch.zeros(num_heads, embed_dim, d_kqv))
        self.Cv = nn.Parameter(torch.zeros(band_num, band_num))

        self.Wq2 = nn.Parameter(torch.zeros(num_heads, embed_dim, d_kqv))
        self.Wk2 = nn.Parameter(torch.zeros(num_heads, embed_dim, d_kqv))
        self.Wv2 = nn.Parameter(torch.zeros(num_heads, embed_dim, d_kqv))

        self.Wo1 = nn.Parameter(torch.zeros(embed_dim, embed_dim))
        self.Wo2 = nn.Parameter(torch.zeros(embed_dim, embed_dim))
        self.dp = nn.Dropout(p=dropout)
        self.corr_weight = nn.Parameter(torch.Tensor([0.2]))
        self.d_kqv = d_kqv
        self.band_num = band_num
        self.nhead = num_heads
        self.h_weight = nn.Parameter(torch.Tensor([0.8]))

        nn.init.kaiming_uniform_(self.Wq)
        nn.init.kaiming_uniform_(self.Wk)
        nn.init.kaiming_uniform_(self.Wv)
        nn.init.kaiming_uniform_(self.Cv)
        nn.init.kaiming_uniform_(self.Wo1)
        nn.init.kaiming_uniform_(self.Wo2)

        nn.init.kaiming_uniform_(self.Wq2)
        nn.init.kaiming_uniform_(self.Wk2)
        nn.init.kaiming_uniform_(self.Wv2)

    def forward(self, x, corr_map, cali_spec=None):
        batch_size = x.size(0)
        x = x.unsqueeze(1)
        x = x.repeat(1, self.nhead, 1, 1)

        Q2s = torch.matmul(x, self.Wq2).permute(2, 1, 0, 3)
        K2s = torch.matmul(x, self.Wk2).permute(2, 1, 3, 0)
        V2s = torch.matmul(x, self.Wv2).permute(2, 1, 0, 3)
        if cali_spec is None:
            A2s = F.softmax(torch.matmul(Q2s, K2s) / np.sqrt(self.d_kqv), dim=-1)
            heads2 = torch.matmul(A2s, V2s).permute(2, 0, 1, 3)
        else:
            cali_spec = cali_spec.unsqueeze(1)
            cali_spec = cali_spec.repeat(1, self.nhead, 1, 1)
            cali_Ks = torch.matmul(cali_spec, self.Wk2).permute(2, 1, 3, 0)
            cali_Vs = torch.matmul(cali_spec, self.Wv2).permute(2, 1, 0, 3)
            cat_Ks = torch.cat([K2s, cali_Ks], dim=-1)
            cat_Vs = torch.cat([V2s, cali_Vs], dim=-2)
            A2s = torch.matmul(Q2s, cat_Ks) / np.sqrt(self.d_kqv)
            for i in range(batch_size):
                if i > 0:
                    A2s[:, :, i, 0:i] = - torch.inf
                if i + 1 < batch_size:
                    A2s[:, :, i, i + 1:batch_size] = -torch.inf
            A2s = F.softmax(A2s, dim=-1)
            heads2 = torch.matmul(A2s, cat_Vs).permute(2, 0, 1, 3)
        heads2 = torch.reshape(heads2, (heads2.size(0), heads2.size(1), -1))
        heads2 = torch.matmul(heads2, self.Wo2)

        h_ff = heads2.unsqueeze(1)
        h_ff = h_ff.repeat(1, self.nhead, 1, 1)
        Qs = torch.matmul(x + h_ff, self.Wq)
        Ks = torch.matmul(x + h_ff, self.Wk).permute(0, 1, 3, 2)
        Vs = torch.matmul(x + h_ff, self.Wq)
        As = ((1 - self.corr_weight) * torch.matmul(Qs, Ks) / np.sqrt(self.d_kqv) +
              self.corr_weight * torch.matmul(corr_map, self.Cv) / np.sqrt(self.band_num))
        As = F.softmax(As, dim=-1)
        heads = torch.matmul(As, Vs).permute(0, 2, 1, 3)
        heads = torch.reshape(heads, (heads.size(0), heads.size(1), -1))
        heads = torch.matmul(heads, self.Wo1)

        heads_f = (1 - self.h_weight) * heads + self.h_weight * heads2
        res = self.dp(heads_f)
        return res, As


class CalibrationSpectra:
    def __init__(self):
        self.cali_spec = None
        self.x = None
        self.y = None
        self.task = None
        self.spec_n = None

    def get_cali_spec(self):
        return torch.Tensor(self.cali_spec)

    def load_cali_spec(self, cali_spec):
        self.cali_spec = cali_spec

    def chang_seed(self, random_seed):
        x, y, task, spec_n = self.x, self.y, self.task, self.spec_n
        sample_n = np.shape(x)[0]
        seed = random_seed
        if task == 'regression':
            idxes = np.argsort(y, axis=0)
            step = int(sample_n / spec_n)
            idx_mat = idxes[0:step * spec_n].reshape((spec_n, step))
            for i, idx_arr in enumerate(idx_mat):
                np.random.seed(seed)
                np.random.shuffle(idx_arr)
                seed += 1
            select_idxes = idx_mat[:, 0]
        else:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=spec_n, random_state=random_seed)
            select_idxes = sss.split(x, y)[0][1]
        self.cali_spec = x[select_idxes]
        return torch.Tensor(self.cali_spec)

    def select_sepc(self, data: dict, spec_n, task, random_seed=0, special=False):
        x = data['x']
        y = data['y']
        self.x = x
        self.y = y
        self.task = task
        self.spec_n = spec_n
        sample_n = np.shape(x)[0]
        if task == 'regression':
            idxes = np.argsort(y, axis=0)
            step = int(sample_n / spec_n)
            idx_mat = idxes[0:step * spec_n].reshape((spec_n, step))
            for i, idx_arr in enumerate(idx_mat):
                np.random.shuffle(idx_arr)
            select_idxes = idx_mat[:, 0]
        elif special:
            this_label = 0
            select_idxes = [0]
            for i, lab in enumerate(y):
                lab_a = np.argmax(lab)
                if lab_a != this_label:
                    this_label = lab_a
                    select_idxes.append(i)
        else:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=spec_n, random_state=random_seed)
            select_idxes = None
            for tr_idxes, te_idxes in sss.split(x, y):
                select_idxes = te_idxes
                break
        self.cali_spec = x[select_idxes]
