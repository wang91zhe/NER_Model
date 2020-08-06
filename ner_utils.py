import mxnet as mx
import gluonnlp as nlp
import argparse


def str2bool(v):
    """Utility function for parsing boolean in argparse

    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    :param v: value of the argument
    :return:
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_context(gpu_index):
    context = None
    if gpu_index is None or gpu_index == "":
        context = mx.cpu()
    if isinstance(gpu_index, int):
        context = mx.gpu()
    return context

def get_ernie_model(ernie_model, cased, ctx, dropout_prob):
    ernie_dataset_name = "baidu_ernie_uncased"
    ernie_model, vocab = nlp.model.get_model(
        name=ernie_model,
        dataset_name=ernie_dataset_name,
        pretrained=True,
        ctx=ctx,
        use_pooler=False,
        use_decoder=False,
        use_classifier=False,
        dropout=dropout_prob,
        embed_dropout=dropout_prob)
    return ernie_model, vocab


