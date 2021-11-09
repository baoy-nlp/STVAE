import numpy as np
import torch


def get_anneal_values(anneal_function, step, k, x0, decay_ratio=0.001):
    if anneal_function == "fixed":
        return 1.0
    elif anneal_function == 'logistic':
        return float(1 / (1 + np.exp(-k * (step - x0))))
    elif anneal_function == 'sigmoid':
        return float(1 / (1 + np.exp(decay_ratio * (x0 - step))))
    elif anneal_function == 'negative-sigmoid':
        return float(1 / (1 + np.exp(-decay_ratio * (x0 - step))))
    elif anneal_function == 'linear':
        return min(1, step / x0)





class Trick(object):
    """
    tricks used in VAE training:
        - word dropout
        - kl-divergence ratio annealing
    """

    def __init__(self, kl_anneal_funcs='logistic', wd_anneal_funcs='linear', unk=0, pad=1, sos=None, eos=None):
        self.unk = unk
        self.pad = pad
        self.sos = sos if sos is not None else -1
        self.eos = eos if eos is not None else -1
        self.kl_anneal_funcs = kl_anneal_funcs
        self.wd_anneal_funcs = wd_anneal_funcs

    def get_wd_ratio(self, alpha, step, k, x0):
        if step < 0:
            return 0.0
        return alpha * get_anneal_values(self.wd_anneal_funcs, step, k, x0)

    def get_kl_weight(self, step, k, x0):
        if step < 0:
            return 1.0
        return get_anneal_values(self.kl_anneal_funcs, step, k, x0)

    def get_dropped_inputs(self, inputs, dropoutr):
        if dropoutr > 0:
            prob = torch.rand(inputs.size())
            if inputs.is_cuda:
                prob = prob.cuda(inputs.get_device())
            prob[(inputs - self.sos) * (inputs - self.pad) * (inputs - self.eos) == 0] = 1
            masked_inputs = inputs.clone()
            masked_inputs[prob < dropoutr] = self.unk
            return masked_inputs
        return inputs
