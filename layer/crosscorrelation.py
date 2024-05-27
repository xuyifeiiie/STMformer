import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from layer.nodeformer import softmax_kernel_transformation, numerator_gumbel, denominator_gumbel
from utils.utils import construct_adj, construct_adj_optimized
import numpy as np

BIG_CONSTANT = 1e8


def create_products_of_givens_rotations(dim, seed):
    nb_givens_rotations = dim * int(math.ceil(math.log(float(dim))))
    q = np.eye(dim, dim)
    np.random.seed(seed)
    for _ in range(nb_givens_rotations):
        random_angle = math.pi * np.random.uniform()  # 生成一个随机角度，范围在 [0, π) 之间。
        random_indices = np.random.choice(dim, 2)  # 随机选择两个不同的索引，用于确定旋转的平面。
        index_i = min(random_indices[0], random_indices[1])
        index_j = max(random_indices[0], random_indices[1])
        slice_i = q[index_i]
        slice_j = q[index_j]
        new_slice_i = math.cos(random_angle) * slice_i + math.cos(random_angle) * slice_j
        new_slice_j = -math.sin(random_angle) * slice_i + math.cos(random_angle) * slice_j
        q[index_i] = new_slice_i
        q[index_j] = new_slice_j
    return torch.tensor(q, dtype=torch.float32)


def create_projection_matrix(m, d, seed=0, scaling=0, struct_mode=False):
    nb_full_blocks = int(m / d)  # m 30 d 32     m=60 d=64
    block_list = []
    current_seed = seed
    for _ in range(nb_full_blocks):
        torch.manual_seed(current_seed)
        if struct_mode:
            q = create_products_of_givens_rotations(d, current_seed)
        else:
            unstructured_block = torch.randn((d, d))  # 32 32
            # q, _ = torch.qr(unstructured_block)
            q, _ = torch.linalg.qr(unstructured_block)
            q = torch.t(q)  # 转置
        block_list.append(q)
        current_seed += 1
    remaining_rows = m - nb_full_blocks * d
    if remaining_rows > 0:
        torch.manual_seed(current_seed)
        if struct_mode:
            q = create_products_of_givens_rotations(d, current_seed)
        else:
            unstructured_block = torch.randn((d, d))
            # q, _ = torch.qr(unstructured_block)
            q, _ = torch.linalg.qr(unstructured_block)
            q = torch.t(q)
        block_list.append(q[0:remaining_rows])
    final_matrix = torch.vstack(block_list)

    current_seed += 1
    torch.manual_seed(current_seed)
    if scaling == 0:
        multiplier = torch.norm(torch.randn((m, d)), dim=1)
    elif scaling == 1:
        multiplier = torch.sqrt(torch.tensor(float(d))) * torch.ones(m)
    else:
        raise ValueError("Scaling must be one of {0, 1}. Was %s" % scaling)

    return torch.matmul(torch.diag(multiplier), final_matrix)  # md dd -> md


def kernelized_gumbel_softmax(query, key, value, kernel_transformation, projection_matrix=None,
                              K=5, tau=0.2):
    """
    fast computation of all-pair attentive aggregation with linear complexity
    input: query/key/value [B, N, H, D]
    return: updated node emb, attention weight (for computing edge loss)
    B = graph number (always equal to 1 in Node Classification), N = node number, H = head number,
    M = random feature dimension, D = hidden size, K = number of Gumbel sampling
    """
    query = query.float()
    key = key.float()
    value = value.float()
    projection_matrix = projection_matrix.float()
    eps = 1e-8
    # query = query.double()
    # key = key.double()
    # value = value.double()
    # projection_matrix = projection_matrix.double()
    query = query / math.sqrt(tau)  # B N H d
    key = key / math.sqrt(tau)
    query_prime = kernel_transformation(query, True, projection_matrix)  # [B, N, H, M]
    key_prime = kernel_transformation(key, False, projection_matrix)  # [B, N, H, M]
    query_prime = query_prime.permute(1, 0, 2, 3)  # [N, B, H, M]
    key_prime = key_prime.permute(1, 0, 2, 3)  # [N, B, H, M]
    value = value.permute(1, 0, 2, 3)  # [N, B, H, D]

    # compute updated node emb, this step requires O(N)  .exponential_().log() + epsilon
    # [N, B, H, K]
    # gumbels = (-torch.empty(key_prime.shape[:-1] + (K,), memory_format=torch.legacy_contiguous_format)).to(query.device)
    # torch.nn.init.trunc_normal_(gumbels, mean=0.5, std=1, a=0, b=1)

    # eps = 1e-4
    # min_val = 1e-3
    # U = Variable(torch.FloatTensor(*gumbels.shape).uniform_(min_val, 1-min_val), requires_grad=False).float()
    # gumbels = (-torch.log(-torch.log(U + eps) + eps) + eps).float().to(query.device)
    tmp_gumbels = torch.empty(key_prime.shape[:-1] + (K,),
                              memory_format=torch.legacy_contiguous_format, dtype=torch.float64).exponential_() + eps
    gumbels = (-tmp_gumbels.log() + eps).to(query.device)  # [N, B, H, K]

    gumbels = gumbels / tau
    gumbels = gumbels.float()
    # [N, B, H, 1, M] * [N, B, H, K, 1] -> [N, B, H, K, M]
    key_prime = key_prime.float()
    # key_prime = key_prime.double()
    key_t_gumbel = key_prime.unsqueeze(3) * gumbels.exp().unsqueeze(4)  # [N, B, H, K, M]
    key_t_gumbel = key_t_gumbel.float()
    # key_t_gumbel = key_t_gumbel.double()

    query_prime = query_prime.float()
    # query_prime = query_prime.double()
    z_num = numerator_gumbel(query_prime, key_t_gumbel, value)  # [N, B, H, K, D]
    z_num = z_num.float()
    # z_num = z_num.double()
    z_den = denominator_gumbel(query_prime, key_t_gumbel) + eps  # [N, B, H, K]
    z_den = z_den.float()
    # z_den = z_den.double()

    z_num = z_num.permute(1, 0, 2, 3, 4)  # [B, N, H, K, D]
    z_den = z_den.permute(1, 0, 2, 3)  # [B, N, H, K]
    z_den = torch.unsqueeze(z_den, len(z_den.shape))  # [B, N, H, K, 1]
    # print("z_num shape", z_num.shape)  # [2, 1792, 8, 10, 64]
    # print("z_den shape", z_den.shape)  # [2, 1792, 8, 10, 1]
    z_output = torch.mean(z_num / z_den, dim=3)  # [B, N, H, D]

    return z_output


def add_conv_relational_bias_adj(x, adj, b, trans='sigmoid'):
    '''
    Compute updated result by the relational bias of input adjacency using adjacency matrix.
    :param x: B L H D
    :param adj: B L L
    '''
    B, L, H, D = x.shape
    adj_norm = F.normalize(adj, p=1, dim=2)
    output = torch.matmul(adj_norm, x.view(B, L, -1)).view(B, L, H, D)
    return output


class Adj_Patch(nn.Module):
    def __init__(self, patch_len, stride, padding):
        super(Adj_Patch, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding = padding
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.padding))

    def forward(self, adj):
        """

        :param adj: B T N N
        :return:
        """
        B, T, N, _ = adj.shape
        adj = adj.reshape(B, T, -1).permute(0, 2, 1)  # B NN T
        adj = self.padding_patch_layer(adj)
        adj = adj.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # B NN L -> B NN num_patch patch_len
        adj = adj.permute(0, 2, 3, 1).reshape(B, -1, self.patch_len, N, N)  # B num_patch patch_len N N
        return adj


class AdjAttention(nn.Module):
    def __init__(self, num_instances, hidden_dim):
        super(AdjAttention, self).__init__()
        self.fc = nn.Linear(num_instances * num_instances, int(hidden_dim / 2))
        self.relu = nn.ReLU()
        self.out = nn.Linear(int(hidden_dim / 2), 1)

    def forward(self, adj):
        """

        :param adj: B*num_patch patch_len N*N
        :return: B*num_patch patch_len 1
        """
        # B*num_patch patch_len N*N -> B*num_patch patch_len hidden_dim/2
        adj = self.relu(self.fc(adj))
        # B*num_patch patch_len hidden_dim/2 -> B*num_patch patch_len 1
        attention_weights = torch.softmax(self.out(adj), dim=1)
        return attention_weights


class PatchAdjFusion(nn.Module):
    def __init__(self, num_instances, hidden_dim):
        super(PatchAdjFusion, self).__init__()
        self.num_instances = num_instances
        self.hidden_dim = hidden_dim
        self.adj_time_attn = AdjAttention(self.num_instances, self.hidden_dim)

    def forward(self, adj_patch):
        """
        B num_patch patch_len N N -> B num_patch N N
        :param adj_patch: B num_patch patch_len N N
        :return: B num_patch N N
        """
        B, num_patch, patch_len, N, _ = adj_patch.shape
        # B num_patch patch_len N N -> B*num_patch patch_len N*N
        reshaped_adj_patch = adj_patch.reshape(-1, patch_len, N * N)
        # B*num_patch patch_len N*N -> B*num_patch patch_len 1
        attn_weights = self.adj_time_attn(reshaped_adj_patch)
        attn_weights = attn_weights.view(B, num_patch, patch_len, 1, 1).to(adj_patch.device)
        # B num_patch patch_len N N  * B num_patch patch_len 1 1 -> B num_patch N N
        fused_adj = torch.sum(adj_patch * attn_weights, dim=2)
        return fused_adj


class CrossCorrealtion(nn.Module):
    def __init__(self, d_model, hidden_dim, n_heads, num_instances, history_steps, patch_len, stride, padding,
                 nb_random_features=64, nb_gumbel_sample=64, tau=0.25, rb_order=1, rb_trans='sigmoid',
                 use_edge_loss=False):
        super(CrossCorrealtion, self).__init__()

        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.num_instances = num_instances

        self.patch_len = patch_len
        self.stride = stride
        self.padding = padding

        self.kernel_transformation = softmax_kernel_transformation
        self.nb_random_features = nb_random_features
        self.nb_gumbel_sample = nb_gumbel_sample
        self.tau = tau
        self.rb_order = rb_order
        self.rb_trans = rb_trans
        self.use_edge_loss = use_edge_loss

        self.d_keys = d_model // n_heads

        if self.rb_order >= 1:
            self.b = torch.nn.Parameter(torch.FloatTensor(self.rb_order, self.n_heads), requires_grad=True)

        self.adj_patch = Adj_Patch(self.patch_len, self.stride, self.padding)
        self.adj_fustion = PatchAdjFusion(self.num_instances, self.hidden_dim)

        self.Wk = nn.Linear(self.d_model, self.d_keys * n_heads)
        self.Wq = nn.Linear(self.d_model, self.d_keys * n_heads)
        self.Wv = nn.Linear(self.d_model, self.d_keys * n_heads)
        self.Wo = nn.Linear(self.d_keys * n_heads, self.d_model)

        self.layernorm = nn.LayerNorm(self.d_keys)

    def forward(self, x, adj):
        """

        :param x: B T' N d_model
        :param adj: B T N N
        :return:
        """
        B, L, N, D = x.shape

        # B T'N H dim,  B L H d
        # B T' N d_model -> B T' N H*d -> B T' N H d -> B T'N H d
        query = self.Wq(x).reshape(B, -1, N, self.n_heads, self.d_keys).reshape(B, -1, self.n_heads, self.d_keys)
        key = self.Wk(x).reshape(B, -1, N, self.n_heads, self.d_keys).reshape(B, -1, self.n_heads, self.d_keys)
        value = self.Wv(x).reshape(B, -1, N, self.n_heads, self.d_keys).reshape(B, -1, self.n_heads, self.d_keys)
        # print("query shape after Wq", query.shape)  # 2, 1792, 8, 64
        seed = torch.ceil(torch.abs(torch.sum(query) * BIG_CONSTANT)).to(torch.int32)
        # M d
        projection_matrix = create_projection_matrix(self.nb_random_features, self.d_keys, seed=seed).to(query.device)

        # B L H d, B E H
        z_next = kernelized_gumbel_softmax(query, key, value, self.kernel_transformation, projection_matrix,
                                           self.nb_gumbel_sample, self.tau)

        z_next = self.layernorm(z_next)
        # print("z_next shape after gumbel", z_next.shape)  # [2, 1792, 8, 64]  1792 = N * L

        # B num_patch patch_len N N
        adj_patch = self.adj_patch(adj).to(x.device)
        # B num_patch patch_len N N -> B num_patch N N
        adj_patch = self.adj_fustion(adj_patch).to(x.device)
        # start = time.time()
        # adj_patch = construct_adj(adj_patch).to(x.device)
        adj_global = construct_adj_optimized(adj_patch).to(x.device)
        # end = time.time()
        # print("construct_adj compute time is {}s".format(end-start))

        for i in range(self.rb_order):
            z_next += add_conv_relational_bias_adj(value, adj_global, self.b[i], self.rb_trans)

        z_next = z_next.float()
        z_next = self.Wo(z_next.flatten(-2, -1))  # B L H d -> B L d_model
        # print("z_next shape after Wo flatten", z_next.shape)  # [2, 1792, 512]
        z_next = z_next.reshape(B, -1, N, self.d_model)

        # if self.use_edge_loss:  # compute edge regularization loss on input adjacency
        #     row, col = edge_index
        #     d_in = degree(col, query.shape[1]).float()
        #     d_norm = 1. / d_in[col]
        #     d_norm_ = d_norm.reshape(1, -1, 1).repeat(1, 1, weight.shape[-1])
        #     link_loss = torch.mean(weight.log() * d_norm_)
        #
        #     z_next = z_next.reshape(B, -1, N, self.d_model)
        #
        #     return z_next, link_loss
        # else:
        #     z_next = z_next.reshape(B, -1, N, self.d_model)
        #     return z_next
        return z_next
