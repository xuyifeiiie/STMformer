import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask


class FullAttention(nn.Module):
    def __init__(self, d_model, n_heads, mask_flag = False, factor = 5, scale = None, attention_dropout = 0.1,
                 output_attention = False):
        super(FullAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attention_mask):
        '''
        :param queries: B T N H d_k
        :param keys: B T S H d_k
        :param values: B T S H d_v
        :param attention_mask: B T 1 N N  上三角矩阵为1，对角线为0，
        :return: B T N H d_v, B T H N S
        '''
        B, T, N, H, d_k = queries.shape
        _, _, S, _, d_v = values.shape
        scale = self.scale or 1. / sqrt(d_k)

        # scores B T H N S
        scores = torch.einsum("btnhe,btshe->bthns", queries, keys)
        if self.mask_flag:
            if attention_mask is None:
                # B T 1 N N
                attention_mask = TriangularCausalMask(B, T, N, device = queries.device)

            scores.masked_fill_(attention_mask.mask, -np.inf)

        # A: B T H N S
        A = self.dropout(torch.softmax(scale * scores, dim = -1))
        # V: B T N H d_v
        V = torch.einsum("bthns,btshd->btnhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(self, d_model, n_heads, mask_flag = True, factor = 5, scale = None, attention_dropout = 0.1,
                 output_attention = False):
        super(ProbAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        """
        :param Q: B H T D_Q
        :param K: B H T D_K
        :param sample_k:
        :param n_top:
        :return: B H n_top D_K, B H n_top
        """
        # old Q [B, H, L, D]
        B, H, T_K, D_K = K.shape
        _, _, T_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, T_Q, T_K, D_K)  # B H -1 T_K D_K -> B T H T_Q T_K D_K
        index_sample = torch.randint(T_K, (T_Q, sample_k))  # T_Q sample_k
        K_sample = K_expand[:, :, torch.arange(T_Q).unsqueeze(1), index_sample, :]  # B H T_Q sample_k D_K
        # Q.unsqueeze(-2): B H T_Q 1 D_K; K_sample.transpose(-2, -1): B H T_Q D_K sample_k
        # B H T_Q 1 D_K matmul B H T_Q D_K sample_k -> B H T_Q 1 sample_k -> B H T_Q sample_k
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), T_K)  # B H T_Q
        M_top = M.topk(n_top, sorted = False)[1]  # B H n_top

        # use the reduced Q to calculate Q_K
        # Q_reduce: B H T_Q D_K -> B H n_top D_K
        Q_reduce = Q[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :]  # factor*ln(L_q)

        # Q_reduce: B H n_top D_K; K.transpose(-2, -1): B H D_K T_K -> B H n_top T_K
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, T_Q):
        """
        :param V: B H T D_V
        :param T_Q: T
        :return: B H T_Q D_V
        """
        B, H, T_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim = -2)  # B H D
            contex = V_sum.unsqueeze(-2).expand(B, H, T_Q, V_sum.shape[-1]).clone()  # B H 1 D -> B H T_Q D
        else:  # use mask
            assert (T_Q == T_V)  # requires that T_Q == T_V, i.e. for self-attention only
            contex = V.cumsum(dim = -2)
        return contex

    def _update_context(self, context_in, V, scores, index, T_Q, attn_mask):
        """
        :param context_in: B H T D_V
        :param V: B H T D_V
        :param scores: B H n_top T_K
        :param index: B H n_top
        :param T_Q: T
        :param attn_mask:
        :return:
        """
        B, H, T_V, D_V = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, T_Q, index, scores, device = V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)  # B H n_top T_K

        attn = torch.softmax(scores, dim = -1)  # nn.Softmax(dim=-1)(scores)  # B H n_top T_K

        # context_in[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :]: B H n_top D_V
        # matmul(attn, V): B H n_top T_K * B H T_K D_V -> B H n_top D_V
        context_in[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] \
            = torch.matmul(attn, V).type_as(context_in)

        if self.output_attention:
            attns = (torch.ones([B, H, T_V, T_V]) / T_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask = None):
        """
        :param queries: B T N H d_k
        :param keys: B T S H d_k
        :param values: B T S H d_v
        :param attn_mask: B T 1 N N  上三角矩阵为1，对角线为0，
        :return: B T N H d_v, B T H N S
        """
        B, T, N, H, D = queries.shape
        _, T, N, _, _ = keys.shape

        queries = queries.transpose(3, 2)  # B T N H d_k -> B T H N d_k
        keys = keys.transpose(3, 2)  # B T N H d_k -> B T H N d_k
        values = values.transpose(3, 2)  # B T N H d_v -> B T H N d_v

        queries = queries.reshape(B, T, H, -1).transpose(2, 1)  # B T H N d_k -> B T H D_Q -> B H T D_Q
        keys = keys.reshape(B, T, H, -1).transpose(2, 1)  # B T H N d_k -> B T H D_K -> B H T D_K
        values = values.reshape(B, T, H, -1).transpose(2, 1)  # B T H N d_v -> B T H D_V -> B H T D_V

        U_part = self.factor * np.ceil(np.log(T)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(T)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < T else T
        u = u if u < T else T

        # B H n_top T_K, B H n_top
        scores_top, index = self._prob_QK(queries, keys, sample_k = U_part, n_top = u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale

        # get the context
        context = self._get_initial_context(values, T)  # B H T_Q D_V

        # update the context with selected top_k queries
        # context : B H T D_V, attn: B H T T
        context, attn = self._update_context(context, values, scores_top, index, T, attn_mask)

        context_output = context.transpose(2, 1).contiguous()  # B H T D_V -> B T H D_V
        context_output = context_output.reshape(B, T, H, -1, D).transpose(3, 2)  # B T H D_V -> B T H N D -> B T N H D
        return context_output, attn


class AttentionLayer(nn.Module):
    def __init__(self, attention_layer, d_model, n_heads, d_keys = None, d_values = None, mix = False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention_layer
        # Q=XWq
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        """
        :param queries: B T N d_model
        :param keys: B T N d_model
        :param values: B T N d_model
        :param attn_mask: Default None
        :return: B T N H d_value, B T H N S
        """
        B, T, N, _ = queries.shape
        _, _, S, _ = keys.shape
        H = self.n_heads

        # B T N d_model -> B T N d_keys * n_heads -> B T N n_heads d_keys
        queries = self.query_projection(queries).view(B, T, N, H, -1)
        # B T S d_model -> B T S d_keys * n_heads -> B T S n_heads d_keys
        keys = self.key_projection(keys).view(B, T, S, H, -1)
        # B T S d_model -> B T S d_values * n_heads -> B T S n_heads d_values
        values = self.value_projection(values).view(B, T, S, H, -1)

        # out B T N H d_value
        # attn B T H N S
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(3, 2).contiguous()  # B T N H d_value -> B T H N d_value
        out = out.view(B, T, N, -1)  # B T N (H*d_value)

        return self.out_projection(out), attn  # B T N*d_model, B T H N S
