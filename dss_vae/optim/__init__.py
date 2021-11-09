from .metric_lr_schedule import MetricLRScheduler

scheduler_cls_dict = {
    'metric': MetricLRScheduler
}


def build_scheduler(args, optimizer):
    cls = scheduler_cls_dict[args.scheduler_cls]
    return cls(args, optimizer)


def load_scheduler(args, fname, optimizer):
    cls = scheduler_cls_dict[args.scheduler_cls]
    return cls.load(fname, optimizer)


__all__ = [
    'MetricLRScheduler',
    'build_scheduler',
    'load_scheduler'
]
