"""
    process: config file
"""
from argparse import Namespace

import yaml


def dict_to_args(dicts):
    return Namespace(**dicts)


def args_to_dict(args):
    dicts = vars(args)
    return dicts


def yaml_to_dict(infile):
    with open(infile, 'r', encoding='utf-8') as f:
        args = yaml.safe_load(f.read())
    return args


def dict_to_yaml(outfile, dicts):
    with open(outfile, "w") as f:
        yaml.dump(dicts, f, default_flow_style=False)


def args_to_yaml(outfile, args):
    dict_to_yaml(outfile=outfile, dicts=args_to_dict(args))


def yaml_to_args(infile):
    dicts = yaml_to_dict(infile=infile)
    args = dict_to_args(dicts)
    return args


def concatenate_dicts(*dicts):
    ret_dict = {}
    for dt in dicts:
        for key, val in dt.items():
            ret_dict[key] = val

    return ret_dict


def yamls_to_args(*infiles):
    dicts = [yaml_to_dict(infile) for infile in infiles]
    dicts = concatenate_dicts(*dicts)
    return dict_to_args(dicts)
