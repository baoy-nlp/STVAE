import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

from .syntax_vae import SyntaxVAE, SequencePredictor, BowPredictor
from .trainer import GenericVAETrainer


class SyntaxVAEAdversary(nn.Module):
    def __init__(self, svae: SyntaxVAE, share_out=False):
        super().__init__()
        args = svae.args

        self.adv_sem_weight = getattr(args, "adv_sem", 0.)
        self.adv_syn_weight = getattr(args, "adv_syn", 0.)
        self.rec_sem_weight = getattr(args, "rec_sem", 0.)
        self.rec_syn_weight = getattr(args, "rec_syn", 0.)
        self.pad_id = svae.pad_id
        self.syn_pad_id = svae.pad_id

        # sem to syn
        if self.adv_sem_weight > 0:
            self.adv_sem = SequencePredictor.build_predictor(
                input_dim=svae.sem_latent_dim,
                embed_dim=svae.syn_embed_dim,
                hidden_dim=svae.syn_hidden_dim,
                sos_id=svae.syn_sos_id,
                eos_id=svae.syn_eos_id,
                pad_id=svae.syn_pad_id,
                num_tokens=svae.multi_syn.embedding.num_embeddings,
                share_input_output_embed=getattr(args, "syn_share_input_output_embed", args.share_input_output_embed),
                output_without_embed=getattr(args, "syn_output_without_embed", args.output_without_embed),
                rnn_type=getattr(args, "syn_rnn_type", args.rnn_type),
                dropout=getattr(args, "syn_dropout", args.dropout),
                out=svae.multi_syn.embedding if share_out else None
            )

        # syn to sem
        if self.adv_syn_weight > 0:
            self.adv_syn = BowPredictor(
                input_dim=svae.syn_latent_dim,
                hidden_dim=args.hidden_dim,
                embed_dim=svae.decoder.embed_dim,
                num_tokens=svae.decoder.num_tokens,
                dropout=args.dropout,
                out=svae.decoder.embedding if share_out else None
            )

        # sem to sents
        if self.rec_sem_weight > 0.:
            self.rec_sem = SequencePredictor.build_predictor(
                input_dim=svae.sem_latent_dim,
                embed_dim=svae.decoder.embed_dim,
                hidden_dim=args.hidden_dim,
                sos_id=svae.sos_id,
                eos_id=svae.eos_id,
                pad_id=svae.pad_id,
                num_tokens=svae.decoder.num_tokens,
                share_input_output_embed=args.share_input_output_embed,
                output_without_embed=args.output_without_embed,
                rnn_type=args.rnn_type,
                dropout=args.dropout,
                out=svae.decoder.embedding if share_out else None
            )

        # syn to sents
        if self.rec_syn_weight > 0.:
            self.rec_syn = SequencePredictor.build_predictor(
                input_dim=svae.syn_latent_dim,
                embed_dim=svae.decoder.embed_dim,
                hidden_dim=args.hidden_dim,
                sos_id=svae.sos_id,
                eos_id=svae.eos_id,
                pad_id=svae.pad_id,
                num_tokens=svae.decoder.num_tokens,
                share_input_output_embed=args.share_input_output_embed,
                output_without_embed=args.output_without_embed,
                rnn_type=args.rnn_type,
                dropout=args.dropout,
                out=svae.decoder.embedding if share_out else None
            )

    @property
    def weights(self):
        return self.adv_sem_weight, self.adv_syn_weight, self.rec_sem_weight, self.rec_syn_weight

    def forward(self, sem_z, syn_z, inputs, syn_inputs, weights=None, adv=True):
        batch_size = inputs.size(1)

        if weights is None:
            weights = self.weights
        semz = sem_z['z']
        synz = syn_z['z']
        if not adv:
            semz = semz.detach()
            synz = synz.detach()

        if weights[0] > 0:
            adv_sem = self.adv_sem(inputs=syn_inputs, hidden=semz, pad_id=self.syn_pad_id) * weights[0]
        else:
            adv_sem = 0.0

        if weights[1] > 0:
            adv_syn = self.adv_syn(inputs=synz, targets=inputs[1:, :]) * weights[1]
        else:
            adv_syn = 0.0

        if weights[2] > 0:
            rec_sem = self.rec_sem(inputs=inputs, hidden=semz, pad_id=self.pad_id) * weights[2]
        else:
            rec_sem = 0.0

        if weights[3] > 0:
            rec_syn = self.rec_syn.forward(inputs=inputs, hidden=synz, pad_id=self.pad_id) * weights[3]
        else:
            rec_syn = 0.0

        losses = [adv_sem / batch_size, adv_syn / batch_size, rec_sem / batch_size, rec_syn / batch_size]
        if adv:
            return tuple(losses)
        else:
            return sum(losses)


class SyntaxVAETrainer(GenericVAETrainer):
    def __init__(self, args, model: SyntaxVAE):
        super().__init__(args, model)

        self.mul_sem_weight = getattr(args, "mul_sem", 0.0)
        self.mul_syn_weight = getattr(args, "mul_syn", 0.0)

        # balance the kl-factor of semantics and syntax
        kl_sem_factor = getattr(args, "kl_sem_factor", 1.0)
        kl_syn_factor = getattr(args, "kl_syn_factor", 1.0)
        self.kl_sem_factor = kl_sem_factor / (kl_sem_factor + kl_syn_factor)
        self.kl_syn_factor = 1 - self.kl_sem_factor

        # adversary for semantic and syntax
        self.adversary = SyntaxVAEAdversary(svae=model, share_out=getattr(args, "share_mul_adv_out", False))
        self.adv_optimizer = self.build_optimizer(self.adversary) if sum(self.adversary.weights) > 0. else None

    def set_optimizer(self, optimizer: Optimizer):
        super().set_optimizer(optimizer)
        if self.adv_optimizer is not None:
            super().set_optimizer(self.adv_optimizer)

    def update_step(self, model, loss, optimizer):
        optimizer.zero_grad()
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad)
        loss.backward()
        optimizer.step()

    def forward(self, model: SyntaxVAE, inputs, optimizer=None, **kwargs):
        features, forward_ret = self.extract_features(
            model, inputs=inputs.src, syn_inputs=inputs.syn, verbose=True, **kwargs
        )

        if self.adv_optimizer is not None:
            # train the adversary
            adv_train_loss = self.adversary.forward(
                sem_z=forward_ret['sem-z'],
                syn_z=forward_ret['syn-z'],
                inputs=inputs.src,
                syn_inputs=inputs.syn,
                adv=False
            )
            if optimizer is not None:
                self.update_step(model=self.adversary, loss=adv_train_loss, optimizer=self.adv_optimizer)
            features["ADV/Loss"] = adv_train_loss.item()

        if optimizer is not None:
            # train general models
            self.update_step(model=model, loss=features['Loss'], optimizer=optimizer)

            # count steps
            self.step += 1

        features["Loss"] = features["Loss"].item()

        return features

    def extract_features(self, model: SyntaxVAE, inputs, syn_inputs=None, verbose=False, **kwargs):
        unk_rate = self.unk_rate
        forward_ret = model(inputs, unk_rate, syn_inputs, **kwargs)
        batch_size = inputs.size(1)

        # NLL-Loss
        logits = forward_ret['logits']
        target = inputs[1:, :]
        nll_loss = self.word_criterion(logits.view(-1, logits.size(-1)), target.view(-1)) / batch_size

        # Multi-Task Loss
        multi_sem_loss = forward_ret['multi-sem'] / batch_size
        multi_syn_loss = forward_ret['multi-syn'] / batch_size

        # Adversarial Loss
        # Split of Syntax-VAE and Adversary, thus could compute it straightforwardly
        adv_sem, adv_syn, rec_sem, rec_syn = self.adversary.forward(
            sem_z=forward_ret['sem-z'],
            syn_z=forward_ret['syn-z'],
            inputs=inputs,
            syn_inputs=syn_inputs
        )

        # Combine the above losses
        mul_loss = multi_sem_loss * self.mul_sem_weight + multi_syn_loss * self.mul_syn_weight
        adv_loss = adv_sem + adv_syn + rec_sem + rec_syn

        # KL-Loss
        kl_weight = self.kl_factor * self.kl_weight

        ret = {
            "Batch": batch_size,
            "UNK": unk_rate,
            "KL/Weight": kl_weight,
            "NLL": nll_loss.item(),
        }

        kl_loss, ret = self._compute_dist_loss(forward_ret['sem-z'], forward_ret['syn-z'], ret)
        loss = nll_loss + kl_loss * kl_weight + mul_loss - adv_loss

        ret.update({
            "KL/Loss": kl_loss.item(),
            "ELBO": (kl_loss + nll_loss).item(),
            "Loss": loss
        })

        det_dict = {
            "MUL/SEM": multi_sem_loss, "MUL/SYN": multi_syn_loss, "ADV/SEM": adv_sem, "ADV/SYN": adv_syn,
            "ADV/REC_SEM": rec_sem, "ADV/REC_SYN": rec_syn
        }
        for key, val in det_dict.items():
            if not isinstance(val, float):
                ret[key] = val.item()

        return ret, forward_ret

    @property
    def verbose_keys(self):
        """Tensorboard Info"""
        origin = super().verbose_keys
        origin.extend(
            ["KL/SEM", "KL/SYN", "MUL/SEM", "MUL/SYN", "ADV/SEM", "ADV/SYN", "ADV/REC_SEM", "ADV/REC_SYN", "ADV/Loss"]
        )
        return origin

    def _compute_dist_loss(self, sem_z, syn_z, ret):
        sem_loss = self.kl_divergence(mean=sem_z['mean'], logv=sem_z['logv'])
        syn_loss = self.kl_divergence(mean=syn_z['mean'], logv=syn_z['logv'])
        sem_kl, syn_kl = sem_loss * self.kl_sem_factor, syn_loss * self.kl_syn_factor
        batch_size = ret["Batch"]
        kl_loss = (sem_kl + syn_kl) / batch_size

        ret.update({
            "KL/Loss": kl_loss.item(),
            "KL/SEM": sem_kl.item() / batch_size,
            "KL/SYN": syn_kl.item() / batch_size,
        })

        return kl_loss, ret

    def saved_params(self):
        params = super().saved_params()
        params["adv_trainer"] = self.adversary.state_dict()

        if self.adv_optimizer is not None:
            params["adv_optimizer"] = self.adv_optimizer.state_dict()

        return params

    @classmethod
    def build(cls, args, model, params=None):
        criterion: SyntaxVAETrainer = super().build(args, model, params)
        if params is not None:
            criterion.adversary.load_state_dict(params["adv_trainer"])
            if criterion.adv_optimizer is not None:
                criterion.adv_optimizer.load_state_dict(params["adv_optimizer"])
        return criterion


class SyntaxTVAETrainer(SyntaxVAETrainer):
    def __init__(self, args, model):
        super().__init__(args, model)

        self.eps = getattr(args, "eps", 1e-8)

        # diverse loss for t
        self.div_t = getattr(args, "div_t", 0.1)

        # use ema
        self.ema_t = getattr(args, "ema_t", False)

        # gumbel softmax for t in forward pass
        self.tau_t = getattr(args, "tau_t", 1.0)  # gumbel tau
        self.gumbel_hard = getattr(args, "gumbel_h", False)  # whether gumbel is hard

        self.t_factor = getattr(args, "t_factor", 1.0)
        self.s_factor = getattr(args, "s_factor", 1.0)

    def forward(self, model: SyntaxVAE, inputs, optimizer=None, **kwargs):
        return super().forward(model, inputs, optimizer, tau=self.tau_t, hard=self.gumbel_hard, **kwargs)

    @property
    def verbose_keys(self):
        """Tensorboard Info"""
        origin = super().verbose_keys
        origin.extend(["KL/T", ])
        return origin

    def _compute_dist_loss(self, sem_z, syn_z, ret):
        batch_size = ret['Batch']

        # kl loss for semantics
        sem_loss = self.kl_divergence(mean=sem_z['mean'], logv=sem_z['logv'])
        sem_kl = sem_loss * self.kl_sem_factor

        # kl loss for syntactic
        syn_s = syn_z["pos_s"]
        s_loss = self.kl_divergence(mean=syn_s["mean"], logv=syn_s["logv"])

        syn_t = syn_z["pos_t"]  # template
        log_t = F.log_softmax(syn_t["logits"] + self.eps, dim=-1)
        t_loss = (syn_t["prob"] * log_t).sum()

        syn_kl = (s_loss * self.s_factor + t_loss * self.t_factor) * self.kl_syn_factor

        # kl loss
        kl_loss = (sem_loss + s_loss) / batch_size
        ret.update({
            "KL/Loss": kl_loss.item(),
            "KL/SEM": sem_kl.item() / batch_size,
            "KL/SYN": syn_kl.item() / batch_size,
            "KL/T": t_loss.item() / batch_size
        })
        return kl_loss, ret
