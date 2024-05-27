import torch.nn as nn
import torch
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        :param d_model:
        :param max_len:
        """
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        print("position: ", position.shape)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        print("div_term: ", div_term.shape)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)


        pe = pe.unsqueeze(0)
        print("pe: ", pe.shape)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: N*d_model
        :return: N*d_model
        """
        return self.pe[:, :x.size(1)]

class EXM(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(EXM, self).__init__()
        self.posembed = PositionalEmbedding(d_model, max_len)

    def forward(self, x):
        y = self.posembed(x)
        return y

if __name__ == '__main__':
    d_model = 500
    max_len = 5000
    x = torch.zeros(1, 3, 3)
    print("x: ", x.shape)
    model = EXM(
        d_model=d_model,
        max_len=max_len
    )
    y = model(x)

    print("y: ", y.shape)

