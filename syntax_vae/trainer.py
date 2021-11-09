import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer


def weight_anneal(anneal_function, step, k, x0):
    if anneal_function == "fixed":
        return 1.0
    elif anneal_function == 'logistic':
        return float(1 / (1 + np.exp(-k * (step - x0))))
    elif anneal_function == 'sigmoid':
        return float(1 / (1 + np.exp(0.001 * (x0 - step))))
    elif anneal_function == 'negative-sigmoid':
        return float(1 / (1 + np.exp(-0.001 * (x0 - step))))
    elif anneal_function == 'linear':
        return min(1, step / x0)
    elif anneal_function == "random":
        return np.random.random()


class GenericVAETrainer(nn.Module):
    def __init__(self, args, model):
        super().__init__()
        self.args = args
        self.pad_id = model.pad_id
        self.unk_factor = args.unk_factor
        self.unk_x0 = args.unk_x0
        self.unk_k = args.unk_k
        self.unk_anneal_func = args.unk_anneal_func

        self.kl_factor = args.kl_factor
        self.kl_x0 = args.kl_x0
        self.kl_k = args.kl_k
        self.kl_anneal_func = args.kl_anneal_func

        self.maximize = args.maximize
        self.word_criterion = nn.CrossEntropyLoss(ignore_index=self.pad_id, reduction="sum")
        self.lr = args.lr
        self.clip_grad = getattr(args, "clip_grad", None)

        self.patience = getattr(args, "patience", 5)
        self.lr_decay = getattr(args, "lr_decay", 0.75)

        # memory for updating
        self.step = 0
        self.count_down = 0
        self.best_metric = None

    def is_best(self, val):
        """ whether the val is the best checkpoint """
        if self.best_metric is None:
            self.best_metric = val
            return True

        is_better = (self.best_metric < val and self.maximize) or (self.best_metric >= val and not self.maximize)

        if is_better:
            self.best_metric = val

        return is_better

    def build_optimizer(self, model, state_dict=None):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        if state_dict is not None:
            optimizer.load_state_dict(state_dict)
        return optimizer

    def set_optimizer(self, optimizer: Optimizer):
        """ set the learning rate of the optimizer with self.lr """
        for para_group in optimizer.param_groups:
            para_group['lr'] = self.lr

    def update_optimizer(self, is_better, optimizer: Optimizer):
        """ lr schedule: count for the patient """

        if is_better:
            self.count_down = 0
        else:
            self.count_down += 1

        if self.count_down < self.patience:
            return False
        self.lr = self.lr * self.lr_decay
        self.set_optimizer(optimizer)
        self.count_down = 0
        return True

    def forward(self, model, inputs, optimizer=None):
        features: dict = self.extract_features(model, inputs.src)
        if optimizer is not None:
            optimizer.zero_grad()
            if self.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad)
            loss = features["Loss"]
            loss.backward()
            optimizer.step()
            self.step += 1

        features['Loss'] = features['Loss'].item()
        return features

    def extract_features(self, model, inputs):
        unk_rate = self.unk_rate
        forward_ret = model(inputs, unk_rate=unk_rate)

        batch_size = inputs.size(1)
        logits = forward_ret['logits']
        target = inputs[1:, :]
        nll_loss = self.word_criterion(logits.view(-1, logits.size(-1)), target.view(-1)) / batch_size

        kl_weight = self.kl_factor * self.kl_weight
        kl_loss = self.kl_divergence(forward_ret['mean'], forward_ret['logv']) / batch_size

        if self.kl_factor > 0:
            loss = kl_loss * kl_weight + nll_loss
            kl_loss = kl_loss.item()
        else:
            # for deterministic AE
            loss = nll_loss
            kl_loss = 0.

        return {
            "Batch": batch_size,
            "UNK": unk_rate,
            "KL/Weight": kl_weight,
            "KL/Loss": kl_loss,
            "NLL": nll_loss.item(),
            "ELBO": kl_loss + nll_loss.item(),
            "Loss": loss,
        }

    def print_out(self, features):
        pr = ""
        for key in self.print_keys:
            pr += "%s %9.4f, " % (key.replace("/", "_"), features[key])

        pr += "%s %9.7f " % ("Lr", self.lr)

        return pr.strip()

    @property
    def print_keys(self):
        """ Info for Training Epochs """
        return ["Loss", "NLL", "KL/Loss", "KL/Weight"]

    @property
    def verbose_keys(self):
        """ Info for Validation and SummaryWritter"""
        return ["ELBO", "NLL", "KL/Loss"]

    def update_record(self, tracker, features):
        batch_size = features.pop("Batch")
        sofar = tracker.pop("Batch")
        for key in self.verbose_keys:
            val = features.get(key, 0)
            if val > 0:
                if key in tracker:
                    tracker[key] = (val * batch_size + tracker[key] * sofar) / (sofar + batch_size)
                else:
                    tracker[key] = val
        tracker['Batch'] = sofar + batch_size
        return tracker

    def print_record(self, tracker, epoch=-1, step=-1):
        pr = ""
        for key in self.verbose_keys:
            val = tracker.get(key, 0)
            if val > 0:
                pr += " %s %9.4f" % (key, tracker[key])

        if epoch > 0:
            return "Epoch %02d,%s" % (epoch, pr)
        if step > -1:
            return "Step %02d,%s" % (step, pr)

    @classmethod
    def kl_divergence(cls, mean, logv):
        # kl_divergence to gaussian prior
        return -0.5 * ((1 + logv - mean.pow(2) - logv.exp()).sum())  # sentence level

    @property
    def unk_rate(self):
        return self.unk_factor * weight_anneal(self.unk_anneal_func, step=self.step, k=self.unk_k, x0=self.unk_x0)

    @property
    def kl_weight(self):
        return weight_anneal(self.kl_anneal_func, step=self.step, k=self.kl_k, x0=self.kl_x0)

    def saved_params(self):
        params = {
            "args": self.args,
            "pad_id": self.pad_id,
            "step": self.step,
            "count_down": self.count_down,
            "lr": self.lr,
            "best_metric": self.best_metric
        }
        return params

    @classmethod
    def build(cls, args, model, params=None):
        if params is not None:
            args = params['args']
            pad_id = params['pad_id']
            criterion = cls(args, model)
            criterion.step = params["step"]
            criterion.best_metric = params["best_metric"]
            criterion.count_down = params.get("count_down", 0)
            criterion.lr = params.get("lr", args.lr)
        else:
            criterion = cls(args, model)
        return criterion
