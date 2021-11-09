from prev_works.prev_dss_vae import DisentangleVAE
from prev_works.prev_dss_vae import BaseSeq2seq
from prev_works.prev_dss_vae.models.vanilla_vae import VanillaVAE
from prev_works.prev_dss_vae import AutoEncoder
from prev_works.prev_dss_vae.vae.enhance_syntax_vae import EnhanceSyntaxVAE
from prev_works.prev_dss_vae import SyntaxGuideVAE
from prev_works.prev_dss_vae.vae.syntax_vae import SyntaxVAE

MODEL_CLS = {
    'Seq2seq': BaseSeq2seq,
    'AutoEncoder': AutoEncoder,
    'VanillaVAE': VanillaVAE,
    'DisentangleVAE': DisentangleVAE,
    'SyntaxGuideVAE': SyntaxGuideVAE,
    'SyntaxVAE': SyntaxVAE,
    'SyntaxVAE2': EnhanceSyntaxVAE,
}


def init_create_model(model: str, **kwargs):
    if model not in MODEL_CLS:
        raise ValueError(
            "Invalid model class \'{}\' provided. Only {} are supported now.".format(
                model, list(MODEL_CLS.keys())))

    return MODEL_CLS[model](**kwargs)


def load_static_model(model: str, model_path: str):
    if model not in MODEL_CLS:
        raise ValueError(
            "Invalid model class \'{}\' provided. Only {} are supported now.".format(
                model, list(MODEL_CLS.keys())))

    return MODEL_CLS[model].load(model_path)
