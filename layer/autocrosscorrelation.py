import torch


def time_delay_agg_training(self, values, corr):
    """
    SpeedUp version of Autocorrelation (a batch-normalization style design)
    This is for the training phase.
    """
    # values -> B H C T
    # corr -> B H C T
    head = values.shape[1]
    channel = values.shape[2]
    length = values.shape[3]
    # find top k
    top_k = int(self.factor * math.log(length))

    mean_value = torch.mean(torch.mean(corr, dim = 1), dim = 1)  # B T

    index = torch.topk(torch.mean(mean_value, dim = 0), top_k, dim = -1)[1]  # T(top_k)
    weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim = -1)
    # update corr
    tmp_corr = torch.softmax(weights, dim = -1)  # B 1 top_k
    # aggregation
    tmp_values = values
    delays_agg = torch.zeros_like(values).float()
    for i in range(top_k):
        pattern = torch.roll(tmp_values, -int(index[i]), -1)
        delays_agg = delays_agg + pattern * \
                     (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
    return delays_agg

