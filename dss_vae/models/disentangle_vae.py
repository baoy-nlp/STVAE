import torch

from dss_vae.modules import BridgeRNNDecoder, BridgeSoftmax, reflect
from dss_vae.modules import SyntaxVAEConstructor
from .vanilla_vae import VanillaVAE


class DisentangleVAE(VanillaVAE):
    def __init__(self, args, vocab, embed=None, name='SyntaxVAE'):
        sem_vocab = getattr(vocab, 'src', vocab)
        syn_vocab = getattr(vocab, 'tgt', vocab)
        ori_vocab = vocab
        super(DisentangleVAE, self).__init__(args, sem_vocab, embed, name)
        self.latent_constructor = SyntaxVAEConstructor(args)
        self.syn_vocab = syn_vocab
        self.sem_vocab = sem_vocab
        self.vocab = ori_vocab

        half_enc_dim = args.enc_hidden_dim // 2

        self.syn_to_syn = BridgeRNNDecoder.build_module(
            args, syn_vocab,
            enc_dim=half_enc_dim,
            enc_layer=args.enc_num_layers * (2 if args.bidir else 1),
            max_len=args.syn_max_len,
        )

        self.syn_to_sem = BridgeSoftmax(
            vocab_size=len(sem_vocab),
            input_size=self.latent_constructor.output_size,
            hidden_size=args.d_hidden,
            embed=self.encoder.embedding if args.share_embed else None
        )

        self.syn_to_sen = BridgeRNNDecoder.build_module(
            args, sem_vocab,
            enc_dim=half_enc_dim,
            enc_layer=args.enc_num_layers * (2 if args.bidir else 1),
            embed=self.encoder.embedding if args.share_embed else None
        )

        self.sem_to_sem = BridgeSoftmax(
            vocab_size=len(sem_vocab),
            input_size=self.latent_constructor.output_size,
            hidden_size=args.d_hidden,
            embed=self.encoder.embedding if args.share_embed else None
        )
        self.sem_to_syn = BridgeRNNDecoder.build_module(
            args, syn_vocab,
            enc_dim=half_enc_dim,
            enc_layer=args.enc_num_layers * (2 if args.bidir else 1),
            max_len=args.syn_max_len,
            embed=self.syn_to_syn.embedding if args.share_embed else None
        )
        self.sem_to_sen = BridgeRNNDecoder.build_module(
            args, sem_vocab,
            enc_dim=half_enc_dim,
            enc_layer=args.enc_num_layers * (2 if args.bidir else 1),
            embed=self.encoder.embedding if args.share_embed else None
        )
        self.net_cls_dict = {
            'syn_to_syn': self.syn_to_syn,
            'syn_to_sem': self.syn_to_sem,
            'syn_to_sen': self.syn_to_sen,
            'sem_to_syn': self.sem_to_syn,
            'sem_to_sem': self.sem_to_sem,
            'sem_to_sen': self.sem_to_sen,
        }

    def predict_sequence(self, feed_var, net_key, hidden, ratio=1.0):
        """
            common function
        """
        net: BridgeRNNDecoder = self.net_cls_dict[net_key]
        outputs, hidden, ret, states = net(
            feed_var,
            hidden,
            None,
            score_func=reflect,
            forcing_ratio=ratio
        )
        scores = torch.stack(outputs)
        predicts = torch.stack(ret['sequence']).squeeze()
        return scores, predicts

    def predict_bow(self, net_key, hidden, logp=True):
        """
            common function for Bag-of-Word losses.
        """
        net: BridgeSoftmax = self.net_cls_dict[net_key]
        scores = net(hidden, logp)
        return scores

    def mul_forward(self, input_ret, sem_scale=1.0, syn_scale=1.0):
        input_ret['mul_sem'] = self.predict_bow(
            net_key='sem_to_sem',
            hidden=input_ret['sem_hid'],
            logp=True,
        ) if sem_scale > 0 else None
        input_ret['mul_syn'], _ = self.predict_sequence(
            feed_var=input_ret['syn'],
            net_key='syn_to_syn',
            hidden=input_ret['syn_hid'],
        ) if syn_scale > 0.0 else (None, None)
        return input_ret

    def adv_forward(self, input_ret, sem_scale=1.0, syn_scale=1.0, detach=True):
        input_ret['adv_sem'], _ = self.predict_sequence(
            feed_var=input_ret['syn'],
            net_key='sem_to_syn',
            hidden=input_ret['sem_hid'].detach() if detach else input_ret['sem_hid'],
        ) if sem_scale > 0.0 else (None, None)

        input_ret['adv_syn'] = self.predict_bow(
            net_key='syn_to_sem',
            hidden=input_ret['syn_hid'].detach() if detach else input_ret['syn_hid'],
            logp=True
        ) if syn_scale > 0 else None
        return input_ret

    def rec_forward(self, input_ret, sem_scale=1.0, syn_scale=1.0, detach=True):
        input_ret['rec_sem'], _ = self.predict_sequence(
            feed_var=input_ret['src'],
            net_key='sem_to_sen',
            hidden=input_ret['sem_hid'].detach() if detach else input_ret['sem_hid'],
        ) if sem_scale > 0.0 else (None, None)

        input_ret['rec_syn'], _ = self.predict_sequence(
            feed_var=input_ret['src'],
            net_key='syn_to_sen',
            hidden=input_ret['syn_hid'].detach() if detach else input_ret['syn_hid'],
        ) if syn_scale > 0.0 else (None, None)
        return input_ret

    def example_to_input(self, examples, use_tgt=True):
        if not isinstance(examples, list):
            examples = [examples]
        src_batch = [e.src for e in examples]
        srcs = self.sem_vocab.process(
            minibatch=src_batch,
            device=torch.device(self.args.gpu) if self.args.cuda else None
        )
        if not use_tgt:
            syns = None
        else:
            try:
                tgt_batch = [e.tgt for e in examples]
                syns = self.syn_vocab.process(
                    minibatch=tgt_batch,
                    device=torch.device(self.args.gpu) if self.args.cuda else None
                )
            except AttributeError:
                syns = None

        return {
            'src': srcs,
            'syn': syns
        }

    def predict(self, examples, to_word=True):
        inputs = self.example_to_input(examples)
        return self.input_to_predict(inputs['src'], to_word)

    def generating_from_prior(self, batch_size, to_word=True):
        input_ret = {"batch_size": batch_size}
        self.latent_constructor.sample_latent(input_ret, keys='sem', mode='prior')
        self.latent_constructor.sample_latent(input_ret, keys='syn', mode='prior')
        return self.latent_to_predict(input_ret, to_word)

    def generating_from_posterior(self, examples, to_word=True, sample_size=1, keys=None):
        input_var = self.example_to_input(examples)
        input_ret = self.input_to_posterior(input_var['src'], is_sampling=True)
        if keys is not None:
            input_ret['{}_z'.format(keys)] = input_ret["{}_mean".format(keys)]
        return self.latent_to_predict(input_ret, to_word)

    def generating_for_paraphrase(self, examples, to_word=True, keys="sem", mode="map", sample_size=1):
        input_var = self.example_to_input(examples)
        pos_ret = self.input_to_posterior(input_var['src'], is_sampling=True)
        pos_ret["{}_z".format(keys)] = self.latent_constructor.sample_latent(
            input_ret=pos_ret,
            keys=keys,
            mode=mode,
            sample_size=sample_size
        )

        if keys == "sem":
            pos_ret['syn_z'] = pos_ret["syn_mean"]
            if sample_size > 1:
                pos_ret["syn_z"] = pos_ret["syn_z"].unsqueeze(1).expand(-1, sample_size, -1).reshape(
                    -1, pos_ret['syn_z'].size(-1)
                )
        else:
            pos_ret["sem_z"] = pos_ret["sem_mean"]
            if sample_size > 1:
                pos_ret["sem_z"] = pos_ret["sem_z"].unsqueeze(1).expand(-1, sample_size, -1).reshape(
                    -1, pos_ret['sem_z'].size(-1)
                )

        return self.latent_to_predict(pos_ret, to_word)

    def transfer_predict(self, inputs, trans_key='sem', to_word=True):
        src_ret = self.input_to_posterior(input_var=inputs['src'], is_sampling=False)
        tgt_ret = self.input_to_posterior(input_var=inputs['tgt'], is_sampling=False)
        src_ret["{}_z".format(trans_key)] = tgt_ret["{}_z".format(trans_key)]
        return self.latent_to_predict(input_ret=src_ret, to_word=to_word)

    def extract_latent(self, examples, use_tgt=False):
        input_var = self.example_to_input(examples, use_tgt=use_tgt)
        input_ret = self.input_to_posterior(input_var['src'], is_sampling=False)
        return {'sem': input_ret['sem_mean'], 'syn': input_ret['syn_mean']}

    def to_word(self, predict_ids):
        return self.modality(predict_ids, self.sem_vocab)
