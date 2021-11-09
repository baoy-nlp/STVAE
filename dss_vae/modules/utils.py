import torch


def reparameterize(mean, var, is_logv=False, sample_size=1):
    if sample_size > 1:
        mean = mean.contiguous().unsqueeze(1).expand(-1, sample_size, -1).reshape(-1, mean.size(-1))
        var = var.contiguous().unsqueeze(1).expand(-1, sample_size, -1).reshape(-1, var.size(-1))

    if not is_logv:
        std = torch.sqrt(var + 1e-10)
    else:
        std = torch.exp(0.5 * var)

    noise = torch.randn_like(std)
    z = mean + noise * std
    return z
