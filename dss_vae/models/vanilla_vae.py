import os

import torch
import torch.nn as nn

from dss_vae.modules.blocks import RNNEncoder, BridgeRNNDecoder, reflect
from dss_vae.modules.latent import SingleLatentConstructor
from dss_vae.utils.inputs import id2word


class VanillaVAE(nn.Module):
    def __init__(self, args, vocab, embed=None, name='VAE'):
        super().__init__()
        self.args = args
        vocab = getattr(vocab, args.vocab_key_words, vocab)
        self.pad_id = vocab.pad_id
        self.vocab = vocab
        self._name = name

        self.encoder: RNNEncoder = RNNEncoder.build_module(args, vocab, embed)
        self.latent_constructor = SingleLatentConstructor(args)

        word_embed = self.encoder.embedding if args.share_embed else None
        self.decoder = BridgeRNNDecoder.build_module(args, vocab, word_embed)

        self.model_info = self._model_info()

    def _model_info(self):
        _model_info = "{}-{}_Layer-{}-{}_Hid-{}_Emb-{}_Latent".format(
            self._name,
            self.args.num_layers,
            self.args.rnn_cell,
            self.args.d_hidden,
            self.args.embed_size,
            self.args.latent_size
        )
        return _model_info

    def tuning_parameters(self):
        return self.parameters()

    def encode(self, input_var):
        mask = input_var.ne(self.pad_id)
        encoder_output, encoder_hidden = self.encoder(input_var, mask)
        return {
            "outputs": encoder_output,
            "hidden": encoder_hidden,
            'batch_size': mask.size(0)
        }

    def decode(self, input_ret, feed_var, decoder_init, encoder_outputs=None, forcing_ratio=1.0):
        decoder_outputs, decoder_hidden, ret_dict, enc_states = self.decoder(
            feed_var,
            decoder_init,
            encoder_outputs,
            score_func=reflect,
            forcing_ratio=forcing_ratio,
        )
        scores = torch.stack(decoder_outputs)
        input_ret['scores'] = scores
        predict_ids = torch.stack(ret_dict['sequence']).squeeze()
        input_ret['predict_ids'] = predict_ids
        return input_ret

    def forward(self, input_var, feed_var=None, forcing_ratio=1.0):
        # if feed_var is None:
        #     feed_var = input_var
        input_ret = self.encode(input_var)
        input_ret = self.latent_constructor(input_ret, is_sampling=self.training)
        input_ret = self.decode(input_ret=input_ret, feed_var=feed_var, decoder_init=input_ret['decode_init'],
                                encoder_outputs=None, forcing_ratio=forcing_ratio)
        return input_ret

    def save(self, fname):
        if self.args.debug:
            return
        dir_name = os.path.dirname(fname)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        config_path = os.path.join(dir_name, 'info.config')
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                f.write("{}".format(self.model_info))
        vocab_path = os.path.join(dir_name, 'vocab.bin')
        self.vocab.save(vocab_path, verbose=False)
        params = {
            'args': self.args,
            'vocab': vocab_path,
            'state_dict': self.state_dict(),
        }
        torch.save(params, fname)

    @classmethod
    def load(cls, fname, use_cuda=True):
        params = torch.load(fname, map_location=lambda storage, loc: storage)
        from dss_vae.data.vocab import load_vocab
        args = params['args']
        vocab = load_vocab(params['vocab'])
        model = cls(args, vocab)
        model.load_state_dict(params['state_dict'], strict=True)
        if use_cuda and args.cuda and torch.cuda.is_available():
            model = model.cuda(device=torch.device(args.gpu))
        return model

    def example_to_input(self, examples):
        if not isinstance(examples, list):
            examples = [examples]
        batch = [e.src for e in examples]
        return self.vocab.process(
            minibatch=batch,
            device=torch.device(self.args.gpu) if self.args.cuda else None
        )

    # below is used for evaluating
    def input_to_posterior(self, input_var, is_sampling=True):
        """ extract the semantic and syntactic """
        input_ret = self.encode(input_var)
        return self.latent_constructor.map_to_latent(input_ret, is_sampling)

    def latent_to_predict(self, input_ret, to_word=True):
        input_ret = self.latent_constructor.reconstruct(input_ret)
        input_ret = self.decode(
            input_ret=input_ret, feed_var=None, decoder_init=input_ret['decode_init'], forcing_ratio=0.0
        )
        predict_ids = input_ret['predict_ids']
        if to_word:
            return self.to_word(predict_ids)
        return predict_ids

    def latent_to_score(self, input_ret, feed_var=None, forcing_ratio=1.0):
        input_ret = self.latent_constructor.reconstruct(input_ret)
        input_ret = self.decode(
            input_ret=input_ret, feed_var=feed_var, decoder_init=input_ret['decode_init'], forcing_ratio=forcing_ratio
        )
        return input_ret

    def input_to_predict(self, input_var, to_word):
        input_ret = self.forward(input_var, forcing_ratio=0.0)
        predict_ids = input_ret['predict_ids']
        if to_word:
            return self.to_word(predict_ids)
        return predict_ids

    def predict(self, examples, to_word=True):
        input_var = self.example_to_input(examples)
        return self.input_to_predict(input_var, to_word)

    def generating_from_prior(self, batch_size, to_word=True):
        input_ret = {"batch_size": batch_size}
        self.latent_constructor.sample_latent(input_ret, mode="prior")
        return self.latent_to_predict(input_ret, to_word)

    def generating_from_posterior(self, examples, to_word=True, sample_size=1):
        input_var = self.example_to_input(examples)
        input_ret = self.input_to_posterior(input_var, is_sampling=True)
        self.latent_constructor.sample_latent(input_ret, mode="posterior")
        return self.latent_to_predict(input_ret, to_word)

    def generating_for_paraphrase(self, examples, to_word=True, keys="sem", mode="map", sample_size=1):
        input_var = self.example_to_input(examples)
        pos_ret = self.input_to_posterior(input_var, is_sampling=True)
        pos_ret["z"] = self.latent_constructor.sample_latent(pos_ret, mode=mode, beam_size=sample_size)
        return self.latent_to_predict(pos_ret, to_word)

    def extract_latent(self, examples):
        input_var = self.example_to_input(examples)
        input_ret = self.input_to_posterior(input_var, is_sampling=False)
        return input_ret['mean']

    @classmethod
    def modality(cls, predict_ids, vocab):
        final_result = []
        if predict_ids.dim() < 2:
            predict_ids = predict_ids.view(-1, 1)
        example_nums = predict_ids.size(-1)
        for i in range(example_nums):
            hyp = predict_ids[:, i].tolist()
            res = id2word(hyp, vocab)
            final_result.append(" ".join(res))
        return final_result

    def to_word(self, predict_ids):
        return self.modality(predict_ids, self.vocab)
