import torch
import torch.nn.functional as F
import torch.nn as nn

from utils.utils import construct_adj, construct_adj_optimized


class gcn_operation(nn.Module):
    def __init__(self, in_dim, out_dim, num_vertices, activation='GLU'):
        """
        图卷积模块
        :param adj: 邻接图
        :param in_dim: 输入维度
        :param out_dim: 输出维度
        :param num_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        """
        super(gcn_operation, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_vertices = num_vertices
        self.activation = activation

        assert self.activation in {'GLU', 'relu', 'GELU'}

        if self.activation == 'GLU' or self.activation == 'GELU':
            self.FC = nn.Linear(self.in_dim, 2 * self.out_dim, bias=True)
        else:
            self.FC = nn.Linear(self.in_dim, self.out_dim, bias=True)

    def forward(self, x, adj, mask=None):
        """
        :param x: (3*N, B, Cin)
        :param adj: (B, 3, N, N)
        :param mask: B 3N 3N
        :return: (3*N, B, Cout)
        """
        # adj = construct_adj(adj)  # B 3N 3N
        adj = construct_adj_optimized(adj)
        # mask[adj != 0] = adj[adj != 0]
        if mask is not None:
            adj = adj.to(mask.device) * mask

        x = torch.einsum('bnm, mbc->nbc', adj.to(x.device), x)  # 3*N, B, Cin

        if self.activation == 'GLU' or self.activation == 'GELU':
            lhs_rhs = self.FC(x)  # 3*N, B, 2*Cout
            lhs, rhs = torch.split(lhs_rhs, self.out_dim, dim=-1)  # 3*N, B, Cout

            out = lhs * torch.sigmoid(rhs)
            del lhs, rhs, lhs_rhs

            return out

        elif self.activation == 'relu':
            return torch.relu(self.FC(x))  # 3*N, B, Cout


class STSGCM(nn.Module):
    def __init__(self, in_dim, out_dims, num_of_vertices, activation='GELU'):
        """
        :param in_dim: 输入维度
        :param out_dims: list 各个图卷积的输出维度
        :param num_of_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        """
        super(STSGCM, self).__init__()
        self.in_dim = in_dim
        self.out_dims = out_dims
        self.num_of_vertices = num_of_vertices
        self.activation = activation

        self.gcn_operations = nn.ModuleList()

        self.gcn_operations.append(
            gcn_operation(
                in_dim=self.in_dim,
                out_dim=self.out_dims[0],
                num_vertices=self.num_of_vertices,
                activation=self.activation
            )
        )

        for i in range(1, len(self.out_dims)):
            self.gcn_operations.append(
                gcn_operation(
                    in_dim=self.out_dims[i - 1],
                    out_dim=self.out_dims[i],
                    num_vertices=self.num_of_vertices,
                    activation=self.activation
                )
            )

    def forward(self, x, adj, mask=None):
        """
        :param x: (3N, B, Cin)
        :param adj: (B, 3, N, N)
        :param mask: B 3N 3N
        :return: (N, B, Cout)
        """
        need_concat = []

        for i in range(len(self.out_dims)):
            x = self.gcn_operations[i](x, adj, mask)
            need_concat.append(x)

        # shape of each element is (1, N, B, Cout)
        need_concat = [
            torch.unsqueeze(
                h[self.num_of_vertices: 2 * self.num_of_vertices], dim=0
            ) for h in need_concat
        ]

        out = torch.max(torch.cat(need_concat, dim=0), dim=0).values  # (N, B, Cout)

        del need_concat

        return out


class STSGCL(nn.Module):
    def __init__(self,
                 history,
                 num_of_vertices,
                 in_dim,
                 out_dims,
                 strides=3,
                 activation='GLU',
                 temporal_emb=True,
                 spatial_emb=True):
        """
        :param history: 输入时间步长
        :param in_dim: 输入维度
        :param out_dims: list 各个图卷积的输出维度
        :param strides: 滑动窗口步长，local时空图使用几个时间步构建的，默认为3
        :param num_of_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        :param temporal_emb: 加入时间位置嵌入向量
        :param spatial_emb: 加入空间位置嵌入向量
        """
        super(STSGCL, self).__init__()
        self.strides = strides
        self.history = history
        self.in_dim = in_dim
        self.out_dims = out_dims
        self.num_of_vertices = num_of_vertices

        self.activation = activation
        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb

        self.STSGCMS = nn.ModuleList()
        for i in range(self.history - self.strides + 1):
            self.STSGCMS.append(
                STSGCM(
                    in_dim=self.in_dim,
                    out_dims=self.out_dims,
                    num_of_vertices=self.num_of_vertices,
                    activation=self.activation
                )
            )

        if self.temporal_emb:
            self.temporal_embedding = nn.Parameter(torch.FloatTensor(1, self.history, 1, self.in_dim))
            # 1, T, 1, Cin

        if self.spatial_emb:
            self.spatial_embedding = nn.Parameter(torch.FloatTensor(1, 1, self.num_of_vertices, self.in_dim))
            # 1, 1, N, Cin

        self.reset()

    def reset(self):
        if self.temporal_emb:
            nn.init.xavier_normal_(self.temporal_embedding, gain=0.0003)

        if self.spatial_emb:
            nn.init.xavier_normal_(self.spatial_embedding, gain=0.0003)

    def forward(self, x, adj, mask=None):
        """
        :param x: B, T, N, Cin
        :param adj: (B, T, N, N)
        :param mask: B 3N 3N
        :return: B, T-2, N, Cout
        """
        if self.temporal_emb:
            x = x + self.temporal_embedding

        if self.spatial_emb:
            x = x + self.spatial_embedding

        need_concat = []
        batch_size = x.shape[0]

        for i in range(self.history - self.strides + 1):
            t = x[:, i: i + self.strides, :, :]  # (B, 3, N, Cin)
            adj_t = adj[:, i: i + self.strides, :, :]  # (B, 3, N, N)
            t = torch.reshape(t, shape=[batch_size, self.strides * self.num_of_vertices, self.in_dim])
            # (B, 3*N, Cin)

            t = self.STSGCMS[i](t.permute(1, 0, 2), adj_t, mask)  # (3*N, B, Cin) -> (N, B, Cout)

            t = torch.unsqueeze(t.permute(1, 0, 2), dim=1)  # (N, B, Cout) -> (B, N, Cout) ->(B, 1, N, Cout)

            need_concat.append(t)

        out = torch.cat(need_concat, dim=1)  # (B, T-2, N, Cout)

        del need_concat, batch_size

        return out


class output_layer(nn.Module):
    def __init__(self, num_of_vertices, history, in_dim, num_features,
                 hidden_dim=64, horizon=1):
        """
        预测层，注意在作者的实验中是对每一个预测时间step做处理的，也即他会令horizon=1
        :param num_of_vertices:节点数
        :param history:输入时间步长
        :param in_dim: 输入维度
        :param hidden_dim:中间层维度
        :param horizon:预测时间步长
        """
        super(output_layer, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.history = history
        self.in_dim = in_dim
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.horizon = horizon

        self.FC1 = nn.Linear(self.in_dim * self.history, self.hidden_dim, bias=True)

        self.FC2 = nn.Linear(self.hidden_dim, self.horizon * self.num_features, bias=True)

    def forward(self, x):
        """
        :param x: (B, Tin, N, Cin)  (B, T - 4, N, Cout)
        :return: (B, Tout, N, F)
        """
        batch_size = x.shape[0]

        x = x.permute(0, 2, 1, 3)  # B, N, Tin, Cin
        # B, N, Tin, Cin -> # B, N, Tin*Cin
        out1 = torch.relu(self.FC1(x.reshape(batch_size, self.num_of_vertices, -1)))
        # (B, N, Tin, Cin) -> (B, N, Tin * Cin) -> (B, N, hidden)

        out2 = self.FC2(out1)  # (B, N, hidden) -> (B, N, horizon)
        out2 = out2.reshape(batch_size, self.num_of_vertices, self.horizon, -1)
        del out1, batch_size

        return out2.permute(0, 2, 1, 3)  # B, horizon, N, F


class STSGCN(nn.Module):
    def __init__(self, settings, device):
        super(STSGCN, self).__init__()
        self.num_of_vertices = settings.num_instances
        self.hidden_dims = settings.hidden_dims
        self.out_layer_dim = settings.num_features
        self.activation = settings.activation
        self.use_mask = settings.use_mask

        self.temporal_emb = settings.temporal_emb
        self.spatial_emb = settings.spatial_emb
        self.history = settings.history_steps
        self.horizon = settings.predict_steps
        self.strides = 3

        self.First_FC = nn.Linear(settings.num_features, settings.d_model, bias=True)
        self.STSGCLS = nn.ModuleList()
        self.STSGCLS.append(
            STSGCL(
                history=settings.history_steps,
                num_of_vertices=self.num_of_vertices,
                in_dim=settings.d_model,
                out_dims=self.hidden_dims[0],  # [64, 64]
                strides=self.strides,
                activation=self.activation,
                temporal_emb=self.temporal_emb,
                spatial_emb=self.spatial_emb
            )
        )

        in_dim = self.hidden_dims[0][-1]  # 64
        history = self.history - (self.strides - 1)  # 62

        for idx, hidden_list in enumerate(self.hidden_dims):
            if idx == 0:
                continue
            self.STSGCLS.append(
                STSGCL(
                    history=history,
                    num_of_vertices=self.num_of_vertices,
                    in_dim=in_dim,
                    out_dims=hidden_list,  # [64, 64]
                    strides=self.strides,
                    activation=self.activation,
                    temporal_emb=self.temporal_emb,
                    spatial_emb=self.spatial_emb
                )
            )

            history -= (self.strides - 1)
            in_dim = hidden_list[-1]  # 64

        self.predictLayer = nn.ModuleList()
        for t in range(self.horizon):
            self.predictLayer.append(
                output_layer(
                    num_of_vertices=self.num_of_vertices,
                    history=history,
                    in_dim=in_dim,
                    num_features=settings.num_features,
                    hidden_dim=settings.hidden_dim,
                    horizon=1
                )
            )

    def forward(self, x_enc, x_dec, x_mark_enc, x_mark_dec, adj_in, adj_out, instances_index_vm, mask=None):
        """
        :param adj_in:  B T N N
        :param x_enc: B, Tin, N, Cin
        :return: B, Tout, N
        """
        self.adj = adj_in  # B T N N
        x = torch.relu(self.First_FC(x_enc))  # B, Tin, N, Cin -> B T N D
        adj_tmp = self.adj[:, 0:self.strides, :, :]  # B 3 N N
        adj_tmp = construct_adj_optimized(adj_tmp)  # B 3N 3N
        if self.use_mask:
            if mask is not None:
                self.mask = mask
            else:
                mask = torch.zeros_like(adj_tmp)
                # mask[self.adj != 0] = self.adj[self.adj != 0]
                self.mask = nn.Parameter(mask)
        else:
            self.mask = None

        for model in self.STSGCLS:
            x = model(x, self.adj, self.mask)
        # every time t-2 (B, T - 2, N, Cout)
        # (B, T - 4, N, Cout)

        need_concat = []
        for i in range(self.horizon):
            out_step = self.predictLayer[i](x)  # (B, 1, N, F)
            need_concat.append(out_step)

        out = torch.cat(need_concat, dim=1)  # B, Tout, N, F

        del need_concat

        return out
