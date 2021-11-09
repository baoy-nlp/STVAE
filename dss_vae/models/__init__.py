import torch

from .mixture_vae import GaussianMixtureVAE
from .syntax_conditional_vae import SyntaxCVAE
from .syntax_gm_vae import SyntaxGMVAE
from .syntax_gmm_vae import SyntaxGMMVAE
from .syntax_tvae import SyntaxTVAE
from .syntax_vae import SyntaxVAE
from .vanilla_vae import VanillaVAE

model_cls_dict = {
    'vae': VanillaVAE,
    'gmm-vae': GaussianMixtureVAE,
    'dss-vae': SyntaxVAE,
    'dss-gmm-vae': SyntaxGMMVAE,
    'syntax-gmvae': SyntaxGMVAE,
    'syntax-tvae': SyntaxTVAE,
    'syntax-cvae': SyntaxCVAE
}


def build_model(args, vocab, **kwargs):
    cls = model_cls_dict[args.model_cls]

    model = cls(args, vocab, **kwargs)
    if args.cuda and torch.cuda.is_available():
        model = model.cuda(device=torch.device(args.gpu))
    return model


def load_model(args, fname):
    print("Load model from {}".format(fname))
    cls = model_cls_dict[args.model_cls]
    return cls.load(fname)


__all__ = [
    'VanillaVAE',
    'GaussianMixtureVAE',
    'SyntaxGMMVAE',
    'SyntaxVAE',
    'SyntaxGMVAE',
    'SyntaxTVAE',
    'SyntaxCVAE',
    'load_model',
    'build_model'
]
