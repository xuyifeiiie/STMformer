import torch.nn as nn


class NormLayer(nn.Module):
    def __init__(self, history_steps, d_model):
        super(NormLayer, self).__init__()
        self.history_steps = history_steps
        self.d_model = d_model
        # self.norm_layer = nn.BatchNorm2d(history_steps)
        self.norm_layer = nn.LayerNorm(self.d_model)

    def forward(self, x_raw, x_computed):
        """
        :param x_raw: B T N d_model
        :param x_computed: B T N d_model
        :return: B T N d_model
        """
        input = x_raw + x_computed
        # input = input.permute(0, 3, 1, 2)
        output = self.norm_layer(input)
        # output = output.permute(0, 2, 3, 1)

        return output


# The last layer of EncoderLayer
class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, after_d_model):
        super(FeedForwardLayer, self).__init__()
        self.linear1 = nn.Linear(d_model, after_d_model)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(after_d_model, d_model)

    def forward(self, inputs):
        """
        :param inputs: B T N d_model
        :return: B T N d_model
        """
        out = self.relu(self.linear1(inputs))
        out = self.linear2(out)
        return out


class DistilFeedForwardLayer(nn.Module):
    def __init__(self, d_model, after_d_model):
        super(DistilFeedForwardLayer, self).__init__()
        self.linear1 = nn.Linear(d_model, 2 * d_model)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2 * d_model, after_d_model)

    def forward(self, inputs):
        """
        :param inputs: B T N d_model
        :return: B T N after_d_model
        """
        out = self.relu(self.linear1(inputs))
        out = self.linear2(out)
        return out


# The last layer of Decoder
class ProjectLayer(nn.Module):
    def __init__(self, d_model, num_features):
        super(ProjectLayer, self).__init__()
        self.linear = nn.Linear(d_model, num_features)

    def forward(self, x):
        """
        :param x: B T N d_model
        :return: B T N num_features
        """
        out = self.linear(x)
        return out


class FlattenPatch(nn.Module):
    def __init__(self, num_patch, history_steps, dropout=0.1):
        super(FlattenPatch, self).__init__()
        self.linear = nn.Linear(num_patch, history_steps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        """

        :param x: B T' N d_model
        :return: B T N d_model
        """
        x = x.permute(0, 2, 3, 1)
        x = self.linear(x)
        x = self.dropout(x)
        x = x.permute(0, 3, 1, 2)
        return x
