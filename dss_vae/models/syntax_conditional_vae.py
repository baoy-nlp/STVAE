from dss_vae.modules.constructor import SyntaxCVAEConstructor
from .syntax_vae import SyntaxVAE


class SyntaxCVAE(SyntaxVAE):
    def __init__(self, args, vocab, embed=None, name='Syntax-Conditional-VAE'):
        super(SyntaxCVAE, self).__init__(args, vocab, embed, name)
        self.latent_constructor = SyntaxCVAEConstructor(args)
