from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from prev_works.prev_dss_vae.decoder import RNNDecoder
from prev_works.prev_dss_vae.encoder import RNNEncoder
from prev_works.prev_dss_vae.models.base_vae import BaseVAE
from prev_works.prev_dss_vae.networks.bridger import MLPBridger
from prev_works.prev_dss_vae.utils.nn_funcs import id2word
from prev_works.prev_dss_vae.utils.nn_funcs import to_input_variable
from prev_works.prev_dss_vae.utils.nn_funcs import to_var
from prev_works.prev_dss_vae.utils.nn_funcs import unk_replace
from prev_works.prev_dss_vae.utils.nn_funcs import wd_anneal_function


class DisentangleVAE(BaseVAE):
    """
    Disentangle the syntax and semantic of VAE's latent spaces.
    Model syntax by predict the syntax-label sequence.
    Model semantic by predict the BoW of src inputs.
    To ensure the success of disentangling, we adopt the adversarial training。


    """

    def score(self, **kwargs):
        pass

    def decode(self, inputs, encoder_outputs, encoder_hidden):
        return self.decoder.forward(
            inputs=inputs,
            encoder_outputs=encoder_outputs,
            encoder_hidden=encoder_hidden
        )

    def __init__(self, args, vocab, src_embed=None, tgt_embed=None):
        super(DisentangleVAE, self).__init__(args, vocab, name="Disentangle VAE with deep encoder")
        print("This is {} with parameter\n{}".format(self.name, self.base_information()))
        if src_embed is None:
            self.src_embed = nn.Embedding(len(vocab.src), args.embed_size)
        else:
            self.src_embed = src_embed
        if tgt_embed is None:
            self.tgt_embed = nn.Embedding(len(vocab.tgt), args.embed_size)
        else:
            self.tgt_embed = tgt_embed

        self.pad_idx = vocab.src.pad_id  # TODO: BUG FIX [2019-07-21]

        self.latent_size = int(args.latent_size)
        self.rnn_type = args.rnn_type
        self.unk_rate = args.unk_rate
        self.step_unk_rate = 0.0
        self.direction_num = 2 if args.bidirectional else 1

        self.enc_hidden_dim = args.enc_hidden_dim
        self.enc_layer_dim = args.enc_hidden_dim * self.direction_num
        self.enc_hidden_factor = self.direction_num * args.enc_num_layers
        self.dec_hidden_factor = args.dec_num_layers
        args.use_attention = False

        if args.mapper_type == "link":
            self.dec_layer_dim = self.enc_layer_dim
        elif args.use_attention:
            self.dec_layer_dim = self.enc_layer_dim
        else:
            self.dec_layer_dim = args.dec_hidden_dim

        var_dim = int(self.enc_hidden_dim * self.enc_hidden_factor / 2)

        # partion dimension from hidden_dims
        task_dim = int(self.enc_layer_dim / 2)
        self.encoder = RNNEncoder(
            vocab_size=len(vocab.src),
            max_len=args.src_max_time_step,
            input_size=args.enc_embed_dim,
            hidden_size=self.enc_hidden_dim,
            embed_droprate=args.enc_ed,
            rnn_droprate=args.enc_rd,
            n_layers=args.enc_num_layers,
            bidirectional=args.bidirectional,
            rnn_cell=args.rnn_type,
            variable_lengths=True,
            embedding=self.src_embed
        )
        # output: [layer*direction ,batch_size, enc_hidden_dim]

        pack_decoder = BridgeRNN(
            args,
            vocab,
            enc_hidden_dim=self.enc_layer_dim,
            dec_hidden_dim=self.dec_layer_dim,
            embed=self.src_embed if args.share_embed else None,
            mode='src'
        )
        self.bridger = pack_decoder.bridger
        self.decoder = pack_decoder.decoder

        if "report" in self.args:
            syn_common = nn.Sequential(
                nn.Linear(var_dim, self.latent_size * 2, True),
                nn.ReLU()
            )
            self.syn_mean = nn.Sequential(
                syn_common,
                nn.Linear(self.latent_size * 2, self.latent_size)
            )
            self.syn_logv = nn.Sequential(
                syn_common,
                nn.Linear(self.latent_size * 2, self.latent_size)
            )

            sem_common = nn.Sequential(
                nn.Linear(var_dim, self.latent_size * 2, True),
                nn.ReLU()
            )
            self.sem_mean = nn.Sequential(
                sem_common,
                nn.Linear(self.latent_size * 2, self.latent_size)
            )
            self.sem_logv = nn.Sequential(
                sem_common,
                nn.Linear(self.latent_size * 2, self.latent_size))
        else:
            self.syn_mean = nn.Linear(var_dim, self.latent_size)
            self.syn_logv = nn.Linear(var_dim, self.latent_size)
            self.sem_mean = nn.Linear(var_dim, self.latent_size)
            self.sem_logv = nn.Linear(var_dim, self.latent_size)

        self.syn_to_h = nn.Linear(self.latent_size, var_dim)
        self.sem_to_h = nn.Linear(self.latent_size, var_dim)

        self.syn_to_syn = BridgeRNN(
            args,
            vocab,
            enc_hidden_dim=task_dim,
            dec_hidden_dim=task_dim,
            embed=tgt_embed,
            mode='tgt'
        )

        self.sem_to_sem = BridgeMLP(
            args=args,
            vocab=vocab,
            enc_dim=task_dim,
            dec_hidden=task_dim,
        )

        self.sem_to_syn = BridgeRNN(
            args,
            vocab,
            enc_hidden_dim=task_dim,
            dec_hidden_dim=task_dim,
            embed=self.tgt_embed if args.share_embed else None,
            mode='tgt'
        )

        self.syn_to_sent = BridgeRNN(
            args,
            vocab,
            enc_hidden_dim=task_dim,
            dec_hidden_dim=task_dim,
            embed=self.src_embed if args.share_embed else None,
            mode='src'
        )

        self.syn_to_sem = BridgeMLP(
            args=args,
            vocab=vocab,
            enc_dim=task_dim,
            dec_hidden=task_dim,
        )

        self.sem_to_sent = BridgeRNN(
            args,
            vocab,
            enc_hidden_dim=task_dim,
            dec_hidden_dim=task_dim,
            embed=self.src_embed if args.share_embed else None,
            mode='src'
        )

    def base_information(self):
        origin = super().base_information()
        return origin + "syn_to_syn:{}\n" \
                        "sem_to_sem:{}\n" \
                        "syn_to_sem:{}\n" \
                        "sem_to_syn:{}\n" \
                        "syn_to_sent:{}\n" \
                        "sem_to_sent:{}\n" \
                        "kl_syn:{}\n" \
                        "kl_sem:{}\n".format(
            str(self.args.mul_syn),
            str(self.args.mul_sem),
            str(self.args.syn_to_sem),
            str(self.args.sem_to_syn),
            str(self.args.syn_to_sent * self.args.infer_weight),
            str(self.args.sem_to_sent * self.args.infer_weight),
            str(self.args.syn_weight),
            str(self.args.sem_weight)
        )

    def encode(self, input_var, length):
        if self.training and self.args.src_wd > 0.:
            input_var = unk_replace(input_var, self.step_unk_rate, self.vocab.src)

        encoder_output, encoder_hidden = self.encoder.forward(input_var, length)
        return encoder_output, encoder_hidden

    def forward(self, examples, is_dis=False):
        if not isinstance(examples, list):
            examples = [examples]
        batch_size = len(examples)

        words = [e.src for e in examples]
        tgt_var = to_input_variable(
            words, self.vocab.src, training=False, cuda=self.args.cuda,
            append_boundary_sym=True, batch_first=True
        )
        syns = [e.tgt for e in examples]
        syn_var = to_input_variable(
            syns, self.vocab.tgt, training=False, cuda=self.args.cuda,
            append_boundary_sym=True, batch_first=True
        )

        ret = self.encode_to_hidden(examples)
        ret = self.hidden_to_latent(ret=ret, is_sampling=self.training)
        ret = self.latent_for_init(ret=ret)
        syn_hidden = ret['syn_hidden']
        sem_hidden = ret['sem_hidden']

        if is_dis:
            # asynthronous training loss
            dis_syn_loss, dis_sem_loss = self.get_dis_loss(
                syn_hidden=syn_hidden,
                sem_hidden=sem_hidden,
                syn_tgt=syn_var,
                sem_tgt=tgt_var
            )
            ret['dis syn'] = dis_syn_loss
            ret['dis sem'] = dis_sem_loss
            return ret

        # reconstruct loss
        decode_init = ret['decode_init']
        sent_decoder_init = self.bridger.forward(decode_init)
        if self.training and self.args.tgt_wd:
            # word dropout
            input_var = unk_replace(tgt_var, self.step_unk_rate, self.vocab.src)
            score = self.decoder.decode(
                inputs=input_var,
                encoder_hidden=sent_decoder_init,
                encoder_outputs=None,
                teacher_forcing_ratio=1.0
            )
        else:
            score = self.decoder.decode(
                inputs=tgt_var,
                encoder_hidden=sent_decoder_init,
                encoder_outputs=None,
            )
        target_var = tgt_var[:, 1:].contiguous()
        _loss = F.cross_entropy(
            input=score.view(-1, score.size(-1)),
            target=target_var.view(-1),
            ignore_index=self.pad_idx,
            reduction='none'
        ).view(tgt_var.size(0), -1)
        reconstruct_loss = (_loss.sum(dim=-1)).view(-1, 1)
        # multi-task loss
        mul_syn_loss, mul_sem_loss = self.get_mul_loss(
            syn_hidden=syn_hidden,
            sem_hidden=sem_hidden,
            syn_tgt=syn_var,
            sem_tgt=tgt_var
        )

        # adversarial loss
        adv_syn_loss, adv_sem_loss = self.get_adv_loss(
            syntax_hidden=syn_hidden,
            semantic_hidden=sem_hidden,
            syn_tgt=syn_var,
            sem_tgt=tgt_var
        )
        ret['adv'] = adv_syn_loss + adv_sem_loss
        ret['mul'] = mul_syn_loss + mul_sem_loss

        ret['nll_loss'] = reconstruct_loss
        ret['sem_loss'] = mul_sem_loss
        ret['syn_loss'] = mul_syn_loss
        ret['batch_size'] = batch_size
        return ret

    def get_loss(self, examples, train_iter, is_dis=False, **kwargs):
        self.step_unk_rate = wd_anneal_function(
            unk_max=self.unk_rate, anneal_function=self.args.unk_schedule,
            step=train_iter, x0=self.args.x0,
            k=self.args.k
        )
        explore = self.forward(examples, is_dis)

        if is_dis:
            return explore

        sem_kl, kl_weight = self.compute_kl_loss(
            mean=explore['sem_mean'],
            logv=explore['sem_logv'],
            step=train_iter,
        )
        syn_kl, _ = self.compute_kl_loss(
            mean=explore['syn_mean'],
            logv=explore['syn_logv'],
            step=train_iter,
        )

        batch_size = explore['batch_size']
        kl_weight *= self.args.kl_factor
        kl_loss = (self.args.sem_weight * sem_kl + self.args.syn_weight * syn_kl) / (
                self.args.sem_weight + self.args.syn_weight)
        kl_loss /= batch_size
        mul_loss = explore['mul'].sum() / batch_size
        adv_loss = explore['adv'].sum() / batch_size
        nll_loss = explore['nll_loss'].sum() / batch_size
        kl_item = kl_loss * kl_weight

        return {
            'KL Loss': kl_loss,
            'NLL Loss': nll_loss,
            'MUL Loss': mul_loss,
            'ADV Loss': adv_loss,
            'KL Weight': kl_weight,
            'KL Item': kl_item,
            'Model Score': kl_loss + nll_loss,
            'ELBO': kl_item + nll_loss,
            'Loss': kl_item + nll_loss + mul_loss - adv_loss,
            'SYN KL Loss': syn_kl / explore['batch_size'],
            'SEM KL Loss': sem_kl / explore['batch_size'],
        }

    def get_adv_loss(self, syntax_hidden, semantic_hidden, syn_tgt, sem_tgt):
        """ Asynchronous adversarial loss """
        args = self.args

        if self.training:
            with torch.no_grad():
                loss_dict = self._dis_loss(syntax_hidden, semantic_hidden, syn_tgt, sem_tgt)
                adv_sem = args.sem_to_syn * loss_dict['sem_to_syn']
                adv_syn = args.syn_to_sem * loss_dict['syn_to_sem']
                if args.infer_weight > 0.:
                    adv_sem += args.infer_weight * args.sem_to_sent * loss_dict['sem_to_sent']
                    adv_syn += args.infer_weight * args.syn_to_sent * loss_dict['syn_to_sent']
                return adv_syn, adv_sem
        else:
            loss_dict = self._dis_loss(syntax_hidden, semantic_hidden, syn_tgt, sem_tgt)
            adv_sem = args.sem_to_syn * loss_dict['sem_to_syn']
            adv_syn = args.syn_to_sem * loss_dict['syn_to_sem']
            if args.infer_weight > 0.:
                adv_sem += args.infer_weight * args.sem_to_sent * loss_dict['sem_to_sent']
                adv_syn += args.infer_weight * args.syn_to_sent * loss_dict['syn_to_sent']
            return adv_syn, adv_sem

    def get_dis_loss(self, syn_hidden, sem_hidden, syn_tgt, sem_tgt):
        """ used in asynchronous forward pass: train the discriminater """
        syn_hidden = syn_hidden.detach()
        sem_hidden = sem_hidden.detach()

        loss_dict = self._dis_loss(syn_hidden, sem_hidden, syn_tgt, sem_tgt)
        if self.args.infer_weight > 0.:
            return loss_dict['sem_to_syn'] + loss_dict['sem_to_sent'], loss_dict['syn_to_sem'] + loss_dict[
                'syn_to_sent']
        else:
            return loss_dict['sem_to_syn'], loss_dict['syn_to_sem']

    def _dis_loss(self, syntax_hidden, semantic_hidden, syn_tgt, sem_tgt):
        sem_to_syn_loss = self.sem_to_syn.forward(hidden=semantic_hidden, tgt_var=syn_tgt)
        syn_to_sem_loss = self.syn_to_sem.forward(hidden=syntax_hidden, tgt_var=sem_tgt)
        if self.args.infer_weight > 0.:
            syn_to_sent_loss = self.syn_to_sent.forward(hidden=syntax_hidden, tgt_var=sem_tgt)
            sem_to_sent_loss = self.sem_to_sent.forward(hidden=semantic_hidden, tgt_var=sem_tgt)
            return {
                'sem_to_syn': sem_to_syn_loss if self.args.sem_to_syn > 0. else 0.,
                'syn_to_sem': syn_to_sem_loss if self.args.syn_to_sem > 0. else 0.,
                'syn_to_sent': syn_to_sent_loss if self.args.syn_to_sent > 0. else 0.,
                "sem_to_sent": sem_to_sent_loss if self.args.sem_to_sent > 0. else 0.
            }
        # 'adv_syn_sup': sem_to_syn_loss if self.args.adv_syn > 0. else 0.,
        # 'adv_sem_sup': syn_to_sem_loss if self.args.adv_sem > 0. else 0.,
        # 'adv_syn_inf': syn_to_sent_loss if self.args.inf_syn > 0. else 0.,
        # "adv_sem_inf": sem_to_sent_loss if self.args.inf_sem > 0. else 0.
        return {
            'sem_to_syn': sem_to_syn_loss,
            'syn_to_sem': syn_to_sem_loss,
        }

    def get_mul_loss(self, syn_hidden, sem_hidden, syn_tgt, sem_tgt):
        syn_loss = self.syn_to_syn.forward(hidden=syn_hidden, tgt_var=syn_tgt)
        sem_loss = self.sem_to_sem.forward(hidden=sem_hidden, tgt_var=sem_tgt)
        return self.args.mul_syn * syn_loss, self.args.mul_sem * sem_loss

    def sample_latent(self, batch_size):
        syntax_latent = to_var(torch.randn([batch_size, self.latent_size]))
        semantic_latent = to_var(torch.randn([batch_size, self.latent_size]))
        return {
            "syn_z": syntax_latent,
            "sem_z": semantic_latent,
        }

    def hidden_to_latent(self, ret, is_sampling=True):
        hidden = ret['hidden']

        def sampling(mean, logv):
            if is_sampling:
                std = torch.exp(0.5 * logv)
                z = to_var(torch.randn([batch_size, self.latent_size]))
                z = z * std + mean
            else:
                z = mean
            return z

        def split_hidden(encode_hidden):
            bs = encode_hidden.size(1)
            factor = encode_hidden.size(0)
            hid = encode_hidden.permute(1, 0, 2).contiguous().view(bs, factor, 2, -1)
            return hid[:, :, 0, :].contiguous().view(bs, -1), hid[:, :, 1, :].contiguous().view(bs, -1)

        batch_size = hidden.size(1)
        sem_hid, syn_hid = split_hidden(hidden)

        semantic_mean = self.sem_mean(sem_hid)
        semantic_logv = self.sem_logv(sem_hid)
        syntax_mean = self.syn_mean(syn_hid)
        syntax_logv = self.syn_logv(syn_hid)
        syntax_latent = sampling(syntax_mean, syntax_logv)
        semantic_latent = sampling(semantic_mean, semantic_logv)

        ret['syn_mean'] = syntax_mean
        ret['syn_logv'] = syntax_logv
        ret['sem_mean'] = semantic_mean
        ret['sem_logv'] = semantic_logv
        ret['syn_z'] = syntax_latent
        ret['sem_z'] = semantic_latent

        return ret

    def latent_for_init(self, ret):
        def reshape(_hidden):
            _hidden = _hidden.view(batch_size, self.enc_hidden_factor, self.enc_hidden_dim // 2)
            _hidden = _hidden.permute(1, 0, 2)
            return _hidden

        syn_z = ret['syn_z']
        sem_z = ret['sem_z']
        batch_size = sem_z.size(0)
        syn_hidden = reshape(self.syn_to_h(syn_z))
        sem_hidden = reshape(self.sem_to_h(sem_z))

        ret['syn_hidden'] = syn_hidden
        ret['sem_hidden'] = sem_hidden
        ret['decode_init'] = torch.cat([syn_hidden, sem_hidden], dim=-1)
        return ret

    def evaluate_(self, examples, beam_size=5):
        if not isinstance(examples, list):
            examples = [examples]
        ret = self.encode_to_hidden(examples)
        ret = self.hidden_to_latent(ret=ret, is_sampling=self.training)
        ret = self.latent_for_init(ret=ret)
        ret['res'] = self.decode_to_sentence(ret=ret)
        return ret

    def predict_syntax(self, hidden, predictor):
        result = predictor.predict(hidden)
        numbers = result.size(1)
        final_result = []
        for i in range(numbers):
            hyp = result[:, i].data.tolist()
            res = id2word(hyp, self.vocab.tgt)
            seems = [[res], [len(res)]]
            final_result.append(seems)
        return final_result

    def extract_variable(self, examples):
        pass

    def eval_syntax(self, examples):
        ret = self.encode_to_hidden(examples, need_sort=True)
        ret = self.hidden_to_latent(ret, is_sampling=False)
        ret = self.latent_for_init(ret)
        return self.predict_syntax(hidden=ret['syn_hidden'], predictor=self.syn_to_syn)

    def eval_adv(self, sem_in, syn_ref):
        """
        adversary predicting
            generating syntax from semantic or syntax.
        Args:
            sem_in:
            syn_ref:

        Returns:

        """
        sem_ret = self.encode_to_hidden(sem_in)
        sem_ret = self.hidden_to_latent(sem_ret, is_sampling=self.training)
        syn_ret = self.encode_to_hidden(syn_ref, need_sort=True)
        syn_ret = self.hidden_to_latent(syn_ret, is_sampling=self.training)
        sem_ret = self.latent_for_init(ret=sem_ret)
        syn_ret = self.latent_for_init(ret=syn_ret)
        ret = dict(sem_z=sem_ret['sem_z'], syn_z=syn_ret['syn_z'])
        ret = self.latent_for_init(ret)
        ret['res'] = self.decode_to_sentence(ret=ret)
        ret['ori syn'] = self.predict_syntax(hidden=sem_ret['syn_hidden'], predictor=self.syn_to_syn)
        ret['ref syn'] = self.predict_syntax(hidden=syn_ret['syn_hidden'], predictor=self.syn_to_syn)
        return ret

    def conditional_generating(self, condition="sem", examples=None):
        """
        generating sentence from partion infos
        Args:
            condition:  sem/sem-only/syn
            examples:
        """
        ref_ret = self.encode_to_hidden(examples)
        ref_ret = self.hidden_to_latent(ref_ret, is_sampling=True)
        if condition.startswith("sem"):
            ref_ret['sem_z'] = ref_ret['sem_mean']
        else:
            ref_ret['syn_z'] = ref_ret['syn_mean']

        if condition == "sem-only":
            sam_ref = self.sample_latent(batch_size=ref_ret['batch_size'])
            ref_ret['syn_z'] = sam_ref['syn_z']

        ret = self.latent_for_init(ret=ref_ret)

        ret['res'] = self.decode_to_sentence(ret=ret)
        return ret


class BridgeRNN(nn.Module):
    def __init__(self, args, vocab, enc_hidden_dim, dec_hidden_dim, embed, mode='src'):
        super().__init__()
        self.pad_id = vocab.tgt.pad_id
        self.bridger = MLPBridger(
            rnn_type=args.rnn_type,
            mapper_type=args.mapper_type,
            encoder_dim=enc_hidden_dim,
            encoder_layer=args.enc_num_layers,
            decoder_dim=dec_hidden_dim,
            decoder_layer=args.dec_num_layers,
        )
        if mode == 'src':
            self.decoder = RNNDecoder(
                vocab=len(vocab.src),
                max_len=args.src_max_time_step,
                input_size=args.dec_embed_dim,
                hidden_size=dec_hidden_dim,
                embed_droprate=args.dec_ed,
                rnn_droprate=args.dec_rd,
                n_layers=args.dec_num_layers,
                rnn_cell=args.rnn_type,
                use_attention=args.use_attention,
                embedding=embed,
                eos_id=vocab.src.eos_id,
                sos_id=vocab.src.sos_id,
            )
        else:
            self.decoder = RNNDecoder(
                vocab=len(vocab.tgt),
                max_len=args.tgt_max_time_step,
                input_size=args.dec_embed_dim,
                hidden_size=dec_hidden_dim,
                embed_droprate=args.dec_ed,
                rnn_droprate=args.dec_rd,
                n_layers=args.dec_num_layers,
                rnn_cell=args.rnn_type,
                use_attention=args.use_attention,
                embedding=embed,
                eos_id=vocab.tgt.eos_id,
                sos_id=vocab.tgt.sos_id,
            )

    def forward(self, hidden, tgt_var):
        decode_init = self.bridger.forward(input_tensor=hidden)
        score = self.decoder.decode(
            inputs=tgt_var,
            encoder_outputs=None,
            encoder_hidden=decode_init,
        )  # batch_size, seq_len, tgt_var
        target_var = tgt_var[:, 1:].contiguous()
        loss = F.cross_entropy(
            input=score.view(-1, score.size(-1)),
            target=target_var.view(-1),
            ignore_index=self.pad_id,
            reduction='none'
        ).view(target_var.size(0), -1)

        return loss.sum(dim=-1).view(-1, 1)

    def predict(self, hidden):
        decode_init = self.bridger.forward(input_tensor=hidden)

        decoder_outputs, decoder_hidden, ret_dict, enc_states = self.decoder.forward(
            inputs=None,
            encoder_outputs=None,
            encoder_hidden=decode_init,
        )
        result = torch.stack(ret_dict['sequence']).squeeze()
        if result.dim() < 2:
            result = result.unsqueeze(1)
        return result


class BridgeMLP(nn.Module):
    def __init__(self, args, vocab, enc_dim, dec_hidden):
        super().__init__()
        self.bridger = MLPBridger(
            rnn_type=args.rnn_type,
            mapper_type=args.mapper_type,
            encoder_dim=enc_dim,
            encoder_layer=args.enc_num_layers,
            decoder_dim=dec_hidden,
            decoder_layer=args.dec_num_layers,
        )
        if "stack_mlp" not in args:
            self.scorer = nn.Sequential(
                nn.Dropout(args.dec_rd),
                nn.Linear(
                    in_features=dec_hidden * args.dec_num_layers,
                    out_features=len(vocab.src),
                    bias=True
                ),
                nn.LogSoftmax(dim=-1)
            )
        else:
            self.scorer = nn.Sequential(
                nn.Dropout(args.dec_rd),
                nn.Linear(
                    in_features=dec_hidden * args.dec_num_layers,
                    out_features=dec_hidden,
                    bias=True
                ),
                nn.ReLU(),
                nn.Linear(
                    in_features=dec_hidden,
                    out_features=len(vocab.src),
                    bias=True
                ),
                nn.LogSoftmax(dim=-1)
            )
        # self.semantic_nll = nn.NLLLoss(ignore_index=vocab.src.pad_id)
        self.pad_id = vocab.src.pad_id

    def forward(self, hidden, tgt_var):
        """

        Args:
            hidden:
            tgt_var:

        Returns: [batch_size,1]

        """
        batch_size = tgt_var.size(0)
        decoder_init = self.bridger.forward(input_tensor=hidden)
        decoder_init = decoder_init.permute(1, 0, 2).contiguous().view(batch_size, -1)
        score = self.scorer.forward(input=decoder_init)  # batch_size, vocab_size
        # sem_loss = bag_of_word_loss(score, tgt_var, self.semantic_nll)
        # TODO: BUG FIX[2019-07-22]
        ignore_token = tgt_var.eq(self.pad_id).float()  # batch_size,vocab_size
        log_score = score.gather(1, tgt_var)  # batch_size, vocab_size
        sem_loss = -(log_score * ignore_token).sum(dim=-1).view(batch_size, 1)
        return sem_loss
