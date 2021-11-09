from dss_vae.modules.latent import GMMLatentConstructor
from .vanilla_vae import VanillaVAE


class GaussianMixtureVAE(VanillaVAE):
    def __init__(self, args, vocab, embed=None, name='GMM-VAE'):
        super(GaussianMixtureVAE, self).__init__(args, vocab, embed, name)
        self.latent_constructor: GMMLatentConstructor = GMMLatentConstructor(args)

        # for gumbel
        self.hard_gumbel = args.hard_gumbel
        self.gumbel_temp = args.init_temp

    def forward(self, input_var, feed_var=None, forcing_ratio=1.0):
        input_ret = self.encode(input_var)
        input_ret = self.latent_constructor(
            input_ret,
            is_sampling=self.training,
            temperature=self.gumbel_temp,
            hard=self.hard_gumbel
        )
        input_ret = self.decode(
            input_ret=input_ret,
            feed_var=feed_var,
            decoder_init=input_ret['decode_init'],
            encoder_outputs=None,
            forcing_ratio=forcing_ratio
        )
        return input_ret
