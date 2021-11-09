import os
import sys

import torch

from dss_vae.model_utils import update_logs
from dss_vae.models import VanillaVAE


def cmp_func_dict(keywords):
    if keywords == 'low':
        return lambda x, y: x < y
    elif keywords == 'high':
        return lambda x, y: x > y


class MetricLRScheduler(object):
    def __init__(self, args, optimizer):
        super(MetricLRScheduler, self).__init__()
        self.args = args
        self.num_trial = 0
        self.patience = 0
        self.metric = None
        self.optimizer = optimizer
        self.cmp_func = cmp_func_dict(args.cmp_func)
        self.tracker = dict()

    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--patience', type=int, default=8)
        parser.add_argument('--reset-optimizer', action='store_true', default=False)
        parser.add_argument('--lr-decay', type=float, default=0.8)
        parser.add_argument('--min-lr', type=float, default=1e-6)
        parser.add_argument('--cmp-func', type=str, default='lt')

    def is_better(self, new_val):
        if self.metric is None or self.cmp_func(new_val, self.metric):
            self.metric = new_val
            return True
        return False

    def step(self, new_val, model: VanillaVAE, model_file, reload_model=False):
        args = self.args
        if self.is_better(new_val):
            self.patience = 0
            print('save model to [%s]' % model_file, file=sys.stdout)
            model.save(model_file)
            self.save(model_file)

        elif self.patience < args.patience:
            self.patience += 1
            print('hit patience %d' % self.patience, file=sys.stdout)

        if self.patience == args.patience:
            self.num_trial += 1
            print('hit #%d trial' % self.num_trial, file=sys.stdout)

            lr = self.optimizer.param_groups[0]['lr'] * args.lr_decay
            print('decay learning rate to %f' % lr, file=sys.stdout)

            if reload_model:
                print('load previously best model', file=sys.stdout)
                model = model.load(model_file)

            print('restore parameters of the optimizers', file=sys.stdout)
            self.optimizer.load_state_dict(MetricLRScheduler.load_optimizer(model_file))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.patience = 0

        lr = self.optimizer.param_groups[0]['lr']
        if lr < self.args.min_lr:
            print('early stop!', file=sys.stdout)

        return model, self.optimizer

    def save(self, model_file_name):
        dir_name = os.path.dirname(model_file_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        optim_path = os.path.join(dir_name, "schedule.bin")
        params = {
            'args': self.args,
            'num_trial': self.num_trial,
            'patience': self.patience,
            'metric': self.metric,
            'optim': self.optimizer.state_dict()
        }
        torch.save(params, optim_path)

    @classmethod
    def load(cls, model_file_name, optimizer):
        dir_name = os.path.dirname(model_file_name)
        optim_path = os.path.join(dir_name, "schedule.bin")
        params = torch.load(optim_path, map_location=lambda storage, loc: storage)
        optimizer.load_state_dict(params['optim'])
        scheduler = cls(params['args'], optimizer)
        scheduler.num_trial = params['num_trial']
        scheduler.patience = params['patience']
        scheduler.metric = params['metric']
        return scheduler

    @classmethod
    def load_optimizer(cls, model_file_name):
        dir_name = os.path.dirname(model_file_name)
        optim_path = os.path.join(dir_name, "schedule.bin")
        params = torch.load(optim_path, map_location=lambda storage, loc: storage)

        return params['optim']

    def update(self, ret_loss):
        self.tracker = update_logs(ret_loss, self.tracker)
        return self.tracker

    def logs(self, show_log_keys=[]):
        log_str = ""
        for key in show_log_keys:
            logs = self.tracker[key]
            if isinstance(logs, torch.Tensor):
                logs = logs.mean().item()
            _log_str = "%s=%.2f  " % (key, logs)
            log_str += _log_str
        return log_str

    def write_logs(self, writer, train_iter, desc="Train"):
        log_dict = self.tracker
        # log for loss
        for key, val in log_dict.items():
            if isinstance(val, torch.Tensor):
                writer.add_scalar(
                    tag="{}/{}".format(desc, key),
                    scalar_value=val.mean().item(),
                    global_step=train_iter
                )
            else:
                writer.add_scalar(
                    tag="{}/{}".format(desc, key),
                    scalar_value=val,
                    global_step=train_iter
                )
        # log for optimizer
        writer.add_scalar(
            tag="Optimize/lr",
            scalar_value=self.optimizer.param_groups[0]['lr'],
            global_step=train_iter
        )
        writer.add_scalar(
            tag='Optimize/trial',
            scalar_value=self.num_trial,
            global_step=train_iter,
        )
        writer.add_scalar(
            tag='Optimize/patience',
            scalar_value=self.patience,
            global_step=train_iter,
        )
