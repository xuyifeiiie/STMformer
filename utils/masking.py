import torch


class TriangularCausalMask:
    def __init__(self, B, T, N, device="cpu"):
        mask_shape = [B, T, 1, N, N]
        with torch.no_grad():
            # torch.triu上三角矩阵，diagonal=1表示对角线为0.
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        """
        :return: B, T, 1, N, N
        """
        return self._mask


class GATMask:
    def __init__(self, B, T, N, adj_mx, device="cpu"):
        """
        :param B:
        :param T:
        :param N:
        :param adj_mx: B TN TN
        :param device:
        :return: B TN TN
        """

        with torch.no_grad():
            adj_mx_bool = adj_mx.bool()  # B TN TN
            self._mask = (~adj_mx_bool).to(device)  # 连接地方是False

    @property
    def mask(self):
        return self._mask


class ProbMask:
    def __init__(self, B, H, T, index, scores, device="cpu"):
        """
        :param B: batch
        :param H: heads
        :param T: time steps
        :param index: B H n_top
        :param scores: B H n_top T_K
        :param device:
        return: mask B H n_top T_K
        """
        _mask = torch.ones(T, scores.shape[-1], dtype=torch.bool).to(device).triu(1)  # T T_K 上三角矩阵，对角线为0

        _mask_ex = _mask[None, None, :].expand(B, H, T, scores.shape[-1])  # B H T T_K

        indicator = _mask_ex[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :].to(
            device)  # B H n_top T_K

        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


class TimeSeriesMask:
    def __init__(self, B, H, T, device="cpu"):
        mask_shape = [B, H, T, T]
        with torch.no_grad():
            # torch.triu上三角矩阵，diagonal=1表示对角线为0.
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask