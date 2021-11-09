from dss_vae.modules.constructor import SyntaxTVAEConstructor
from .syntax_gm_vae import SyntaxGMVAE


class SyntaxTVAE(SyntaxGMVAE):
    def __init__(self, args, vocab, embed=None, name='Syntax-Template-VAE'):
        super(SyntaxTVAE, self).__init__(args, vocab, embed, name)
        self.latent_constructor = SyntaxTVAEConstructor(args)

        self.hard_gumbel = args.hard_gumbel  # used for template
        self.gumbel_temp = args.init_temp

    # def extract_template_codes(self):
    #     return self.latent_constructor.extract_codes()

    def extract_latent(self, examples, use_tgt=False):
        input_var = self.example_to_input(examples, use_tgt=use_tgt)
        input_ret = self.input_to_posterior(input_var['src'], is_sampling=False)
        input_ret['sem'] = input_ret['sem_mean']
        input_ret['syn'] = input_ret['syn_mean']
        return input_ret
