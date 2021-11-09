import os

import torch


def update_logs(ret_loss, tracker):
    tracker = tracker
    for key, val in ret_loss.items():
        if isinstance(val, torch.Tensor):
            if key in tracker:
                tracker[key] = torch.cat((tracker[key], val.mean().unsqueeze(0)))
            else:
                tracker[key] = val.mean().unsqueeze(0)
        else:
            tracker[key] = val
    return tracker


def update_detached_logs(ret_loss, tracker):
    tracker = tracker
    for key, val in ret_loss.items():
        if isinstance(val, torch.Tensor):
            if key in tracker:
                tracker[key] = torch.cat((tracker[key], val.detach().mean().unsqueeze(0)))
            else:
                tracker[key] = val.detach().mean().unsqueeze(0)
        else:
            tracker[key] = val
    return tracker


def write_logs(log_dict, writer, train_iter, desc="Train"):
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


def init_model(args, vocab, **kwargs):
    from dss_vae.models import build_model
    model = build_model(args, vocab, **kwargs)
    for param in model.tuning_parameters():
        param.data.uniform_(-0.08, 0.08)
    print(model.model_info)
    return model


def init_optimizer(args, model):
    if args.optim_type == 'adam':
        params = [param for param in model.parameters() if param.requires_grad]
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=args.betas)
        return optimizer
    else:
        raise NotImplementedError


def is_valid(fname):
    return fname is not None and os.path.exists(fname)


def build_model(args, vocab, model_file=None, **kwargs):
    try:
        from dss_vae.models import load_model
        return load_model(args, model_file)
    except FileNotFoundError:
        print('Prepare Model from scratch！')
        return init_model(args, vocab, **kwargs)


def build_scheduler(args, model, fname=None, **kwargs):
    if not args.mode.startswith('train'):
        return None
    if args.valid_item == 'BLEU':
        args.cmp_func = 'high'
    elif args.valid_item == 'ELBO':
        args.cmp_func = 'low'
    optimizer = init_optimizer(args, model)
    try:
        from dss_vae.optim import load_scheduler
        return load_scheduler(args, fname, optimizer)
    except FileNotFoundError:
        from dss_vae.optim import build_scheduler
        return build_scheduler(args, optimizer)


def init_runtime_env(args):
    exp_root = args.exp_dir if os.path.isabs(args.exp_dir) else os.path.join(args.home, args.exp_dir)
    desc = "{}/{}".format(args.dataset, args.data_desc)

    model_root = os.path.join(exp_root, "model", desc, args.model_cls)
    logdir_root = os.path.join(exp_root, "log", desc, args.model_cls)
    outdir_root = os.path.join(exp_root, "out", desc, args.model_cls)

    if not os.path.exists(model_root) and not args.debug:
        os.makedirs(model_root)
        os.makedirs(logdir_root, exist_ok=True)
        os.makedirs(outdir_root, exist_ok=True)

    exp_desc = getattr(args, 'exp_desc', 'basic')
    model_dir = os.path.join(model_root, exp_desc)
    log_dir = os.path.join(logdir_root, exp_desc)
    out_dir = os.path.join(outdir_root, exp_desc)

    if not os.path.exists(model_dir) and not args.debug:
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

    return {
        "model_file": os.path.join(model_dir, "model.pkl"),
        'model_dir': model_dir,
        'log_dir': log_dir,
        "out_dir": out_dir,
    }
