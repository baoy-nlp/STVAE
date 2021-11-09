from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from dss_vae.data import prepare_dataset
from dss_vae.extract import extract_vae
from dss_vae.model_utils import init_runtime_env, build_model
from dss_vae.search import search_vae, add_search_args
from dss_vae.test import test_vae
from dss_vae.train import train_vae
from dss_vae.utils.configs import yamls_to_args


def re_parse_args(pre_config, pre_args):
    new_parser = argparse.ArgumentParser()
    all_args = vars(pre_config)
    all_args.update(vars(pre_args))
    for name, val in all_args.items():
        new_parser.add_argument(
            '--{}'.format(name.replace("_", "-")),
            type=type(val),
            default=val
        )
    return new_parser.parse_known_args()[0]


if __name__ == "__main__":

    command_lines = " ".join(sys.argv[1:])
    print("Command Line: {}".format(command_lines))

    parser = argparse.ArgumentParser()

    parser.add_argument('--data-configs', type=str)
    parser.add_argument('--model-configs', type=str)
    parser.add_argument('--run-configs', type=str)
    parser.add_argument('--configs', type=str, nargs="*", default=[])
    parser.add_argument('--exp-desc', type=str, default='')
    parser.add_argument('--mode', type=str, default='extract')  # choices=run_mode_dict.keys()
    parser.add_argument('--debug', action='store_true', default=False)

    parser.add_argument('--eval-func', type=str, default='reconstruct')
    parser.add_argument('--eval-dataset', type=str, default='valid')
    parser.add_argument('--sample-size', type=int, default=1)
    parser.add_argument('--sample-mode', type=str, default="posterior")

    parser = add_search_args(parser)
    config = parser.parse_known_args()[0]

    if len(config.configs) > 0:
        args = yamls_to_args(*config.configs)
    else:
        args = yamls_to_args(*[config.data_configs, config.run_configs, config.model_configs])

    # args = re_parse_args(config, args)
    args.mode = config.mode
    print("=" * 30, "\tMode:{}\t".format(args.mode), "=" * 30, file=sys.stderr)

    args.exp_desc = config.exp_desc
    args.debug = config.debug
    args.eval_func = config.eval_func
    args.eval_dataset = config.eval_dataset

    # File Configuration
    dirs_dict = init_runtime_env(args)
    model_file = dirs_dict['model_file']

    # Dataset, Vocab
    print("=" * 30, "\tLoading Data\t", "=" * 30, file=sys.stderr)
    datasets, vocab = prepare_dataset(args)
    print("=" * 30, "\tLoad Data Finish!\t", "=" * 30, file=sys.stderr)

    model = build_model(args, vocab, model_file)

    if args.mode == "train":
        train_vae(
            args=args, file_dict=dirs_dict, model=model, datasets=datasets, rest_dir=dirs_dict['out_dir'],
            sample_size=config.sample_size, mode=config.sample_mode
        )
    elif args.mode == "test":
        test_vae(
            args=args, file_dict=dirs_dict, model=model, datasets=datasets, rest_dir=dirs_dict['out_dir'],
            sample_size=config.sample_size, mode=config.sample_mode
        )
    elif args.mode == "extract":
        extract_vae(
            args=args, file_dict=dirs_dict, model=model, datasets=datasets, rest_dir=dirs_dict['out_dir'],
            sample_size=config.sample_size, mode=config.sample_mode
        )
    elif args.mode == "search":
        search_vae(
            args=args, file_dict=dirs_dict, model=model, datasets=datasets, rest_dir=dirs_dict['out_dir'],
            sample_size=config.sample_size, mode=config.sample_mode
        )
