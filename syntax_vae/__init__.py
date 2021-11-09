import os

import torch

from syntax_vae.dataset import GenericDataset
from syntax_vae.syntax_trainer import SyntaxVAETrainer, GenericVAETrainer, SyntaxTVAETrainer
from syntax_vae.syntax_tvae import SyntaxTVAE, SyntaxVAE
from syntax_vae.vanilla_vae import VAE

MODEL_CLS_DICT = {
    'VAE': VAE,
    "SyntaxVAE": SyntaxVAE,
    "SyntaxTVAE": SyntaxTVAE
}

Trainer_CLS_DICT = {
    "VAE": GenericVAETrainer,
    "SyntaxVAE": SyntaxVAETrainer,
    "SyntaxTVAE": SyntaxTVAETrainer,
}


def load_checkpoint(args, path, dataset, replace_args=False, logger=None):
    if os.path.exists(path):
        params = torch.load(path, map_location=lambda storage, loc: storage)
        load_args = params["args"] if replace_args else args
        model_params, trainer_params, opt_params = params['model'], params['trainer'], params['opt']
        if logger is not None:
            logger.info("Load Checkpoints from {}".format(path))
    else:
        load_args = args
        model_params, trainer_params, opt_params = None, None, None
    MODEL_CLS = MODEL_CLS_DICT[load_args.cls]
    TRAIN_CLS = Trainer_CLS_DICT[load_args.cls]

    model = MODEL_CLS.build(args, dataset.fields, model_params)
    print(str(model))
    criterion = TRAIN_CLS.build(args, model, trainer_params)
    if args.gpu and torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    print(criterion)
    optimizer = criterion.build_optimizer(model, opt_params)
    return load_args, model, criterion, optimizer


def save_checkpoint(args, path, model, criterion, optimizer, logger=None):
    params = {
        "args": args,
        "model": model.saved_params(),
        "trainer": criterion.saved_params(),
        "opt": optimizer.state_dict()
    }
    try:
        torch.save(params, path)
        if logger is not None:
            logger.info("Save Checkpoint to {}".format(path))
        return True
    except Exception:
        raise RuntimeError("save checkpoint error")
