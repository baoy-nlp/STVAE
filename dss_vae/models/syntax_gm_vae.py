from dss_vae.modules.constructor import SyntaxGMVAEConstructor
from .syntax_vae import SyntaxVAE


class SyntaxGMVAE(SyntaxVAE):
    def __init__(self, args, vocab, embed=None, name='Syntax-GM-VAE'):
        super(SyntaxGMVAE, self).__init__(args, vocab, embed, name)
        self.latent_constructor = SyntaxGMVAEConstructor(args)
        self.gumbel = args.hard_gumbel
        self.temp = args.init_temp

    def forward(self, input_var, feed_var=None, forcing_ratio=1.0):
        input_ret = self.encode(input_var)
        input_ret = self.latent_constructor(
            input_ret, is_sampling=self.training, temperature=self.temp, hard=self.gumbel
        )
        input_ret = self.decode(
            input_ret=input_ret, feed_var=feed_var, decoder_init=input_ret['decode_init'],
            encoder_outputs=None, forcing_ratio=forcing_ratio
        )
        return input_ret

    def generating_from_prior(self, batch_size, to_word=True):
        input_ret = {"batch_size": batch_size}
        input_ret['sem_z'] = self.latent_constructor.sample_latent(input_ret, keys='sem')
        input_ret['syn_z'] = self.latent_constructor.sample_latent(input_ret, keys='syn')
        return self.latent_to_predict(input_ret, to_word)

    def generating_from_posterior(self, examples, to_word=True, sample_size=1, keys=None):
        input_var = self.example_to_input(examples)
        input_ret = self.input_to_posterior(input_var['src'], is_sampling=True)
        if keys is not None:
            if keys == 'meta-syn':
                syn_z = self.latent_constructor.sample_latent(input_ret, keys)
                sem_z = input_ret["sem_mean"].unsqueeze(0).expand(self.latent_constructor.num_components, -1, -1)
                input_ret['sem_z'] = sem_z.contiguous().view(-1, self.args.latent_size)
                input_ret['syn_z'] = syn_z.contiguous().view(-1, self.args.latent_size)
                input_ret['batch_size'] = syn_z.size(0)
            else:
                input_ret['{}_z'.format(keys)] = input_ret["{}_mean".format(keys)]
        return self.latent_to_predict(input_ret, to_word)
