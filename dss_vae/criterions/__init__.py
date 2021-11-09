from .gmvae_criterion import GMMVAECriterion
from .syntax_cvae_criterion import SyntaxCVAECriterion
from .syntax_gmvae_criterion import SyntaxGMMVAECriterion
from .syntax_tvae_criterion import SyntaxTVAECriterion
from .syntax_vae_criterion import SyntaxVAECriterion
from .vae_criterion import VAECriterion

criterion_cls_dict = {
    'vae': VAECriterion,
    'dss-vae': SyntaxVAECriterion,
    'dss-gmm-vae': SyntaxGMMVAECriterion,
    'gmm-vae': GMMVAECriterion,
    'syntax-gmvae': SyntaxGMMVAECriterion,
    'syntax-tvae': SyntaxTVAECriterion,
    'syntax-cvae': SyntaxCVAECriterion
}


def build_criterion(args, vocab, **kwargs):
    cls = criterion_cls_dict[args.model_cls]
    criterion = cls(args, vocab, **kwargs)
    return criterion


__all__ = [
    'VAECriterion',
    'SyntaxVAECriterion',
    'SyntaxGMMVAECriterion',
    'GMMVAECriterion',
    'SyntaxTVAECriterion',
    'SyntaxCVAECriterion',
    'build_criterion'
]
