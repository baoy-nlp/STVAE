""" training of VAEs or Syntax-VAEs """
import sys

import gc
import torch
from tensorboardX import SummaryWriter

from dss_vae.criterions import build_criterion
from dss_vae.evaluate import eval_vae
from dss_vae.model_utils import write_logs, build_scheduler
from dss_vae.models import SyntaxVAE


def validate(model, iterator, criterion, step=-1, out_dir='tmp'):
    model.eval()
    ret = eval_vae(iterator, model, criterion, out_dir, step)
    print(
        '\n== [Step %d] BLEU=%.3f ELBO=%.3f took %d s ==' % (step, ret['BLEU'], ret['ELBO'], ret['EVAL TIME']),
        file=sys.stdout
    )
    return ret


def adversary_step(args, model: SyntaxVAE, criterion, input_var, step: int, optimizer):
    def _update(_loss):
        optimizer.zero_grad()
        _loss.backward()
        if args.clip_grad > 0.:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.forward()

    adversary_ret = criterion(model, input_var, step, True)
    _update(adversary_ret['sem_loss'])
    _update(adversary_ret['syn_loss'])


def train_step(args, model, criterion, batch, step, optimizer, scheduler):
    input_var = model.example_to_input(batch)
    if args.adv_training:
        adversary_step(args, model, criterion, input_var, step, optimizer)

    train_num = len(batch)
    optimizer.zero_grad()
    forward_ret = criterion(model, input_var, step)
    scheduler.update(forward_ret)
    batch_loss = forward_ret['Loss']
    loss = batch_loss.mean()
    train_loss = loss.item()

    loss.backward()
    if args.clip_grad > 0.:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

    optimizer.forward()
    return train_loss, train_num


def train_vae(args, file_dict, model, datasets, **kwargs):
    model_file = file_dict['model_file']
    log_dir = file_dict['log_dir']

    writer = SummaryWriter(log_dir)
    writer.add_text('model_info', model.model_info)

    scheduler = build_scheduler(args, model, fname=model_file)
    print("=" * 30, "\tStart Training\t", "=" * 30)
    iterators = datasets['train']
    steps = getattr(args, 'steps', 0)
    optimizer = scheduler.optimizer
    criterion = build_criterion(args=args, vocab=model.vocab)

    train_loss = 0.0
    train_num = 0.0

    for batch in iterators:
        model.train()
        steps += 1
        step_loss, step_num = train_step(args, model, criterion, batch, steps, optimizer, scheduler)
        train_num += step_num
        train_loss += (step_loss * step_num)

        if steps % args.log_every == 0:
            model.eval()
            torch.cuda.empty_cache()
            gc.collect()
            print(
                '\r[Step %d] Train Loss=%.2f  %s' %
                (steps, train_loss / train_num, scheduler.logs(criterion.printable())), file=sys.stdout, end=" "
            )
            scheduler.write_logs(writer, train_iter=steps, desc='Train')

        if steps % args.eval_every == 0 and steps > args.warm_up:
            eval_dict = validate(
                model, datasets['valid'], criterion=criterion, step=steps, out_dir=file_dict['out_dir']
            )
            check_score = eval_dict[args.valid_item]
            model, optimizer = scheduler.forward(new_val=check_score, model=model, model_file=model_file)
            writer.add_scalar(tag='Valid/Best %s' % args.valid_item, scalar_value=scheduler.metric, global_step=steps)
            write_logs(eval_dict, writer, steps, desc='Valid')
