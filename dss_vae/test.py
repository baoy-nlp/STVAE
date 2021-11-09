from dss_vae.evaluate import reconstruct, sampling, paraphrase, meta_paraphrase, \
    transfer


def format_output(val, keep_num=3):
    if isinstance(val, str):
        return val
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        return "%0.{}f".format(keep_num) % val
    if isinstance(val, list) or isinstance(val, tuple):
        return "\t".join([format_output(item) for item in val])


def print_output(eval_ret):
    for key, val in eval_ret.items():
        print("==\t%s: %s\t==" % (key, format_output(val)))


def test_vae(args, model, datasets, **kwargs):

    if args.eval_func == "reconstruct":
        evaluate = reconstruct
    elif args.eval_func == "sampling":
        evaluate = sampling
    elif args.eval_func == "paraphrase":
        evaluate = paraphrase
    elif args.eval_func == "transfer":
        evaluate = transfer
    else:
        evaluate = meta_paraphrase

    model = model.eval()
    gen_set = args.eval_dataset
    if gen_set is not None:
        print("Evaluate on {}".format(args.eval_dataset))
        output = evaluate(iterator=datasets[gen_set], model=model, **kwargs)
        print_output(output)
    else:
        for name, iterator in datasets.items():
            if iterator is not None:
                print("Evaluate on {}".format(name))
                output = evaluate(iterator=iterator, model=model, **kwargs)
                print_output(output)
