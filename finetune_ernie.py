import os
import numpy as np
import random
import argparse
import mxnet as mx

from ner_utils import get_context, str2bool, get_ernie_model
from data import ERNIETaggingDataset
from model import ERNIETagger


def parse_args():
    """Parse command line arguments."""
    arg_parser = argparse.ArgumentParser(
        description='Train a ERNIE-based named entity recognition model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # data file paths
    arg_parser.add_argument('--train-path', type=str, default="./data/train.txt",
                            help='Path to the training data file')
    arg_parser.add_argument('--dev-path', type=str, default="./data/dev.txt",
                            help='Path to the development data file')
    arg_parser.add_argument('--test-path', type=str, default="./data/test.txt",
                            help='Path to the test data file')

    arg_parser.add_argument('--save-checkpoint-prefix', type=str, default="./model/",
                            help='Prefix of model checkpoint file')

    # bert options
    arg_parser.add_argument('--ernie-model', type=str, default='ernie_12_768_12',
                            help='Name of the BERT model')
    arg_parser.add_argument('--cased', type=str2bool, default=True,
                            help='Path to the development data file')
    arg_parser.add_argument('--dropout-prob', type=float, default=0.1,
                            help='Dropout probability for the last layer')

    # optimization parameters
    arg_parser.add_argument('--seed', type=int, default=13531,
                            help='Random number seed.')
    arg_parser.add_argument('--seq-len', type=int, default=256,
                            help='The length of the sequence input to BERT.'
                                 ' An exception will raised if this is not large enough.')
    arg_parser.add_argument('--gpu', type=int, default=0,
                            help='Number (index) of GPU to run on, e.g. 0.  '
                                 'If not specified, uses CPU.')
    arg_parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    arg_parser.add_argument('--num-epochs', type=int, default=4, help='Number of epochs to train')
    arg_parser.add_argument('--optimizer', type=str, default='bertadam',
                            help='Optimization algorithm to use')
    arg_parser.add_argument('--learning-rate', type=float, default=5e-5,
                            help='Learning rate for optimization')
    arg_parser.add_argument('--warmup-ratio', type=float, default=0.1,
                            help='Warmup ratio for learning rate scheduling')
    args = arg_parser.parse_args()
    return args

def main(config):
    ctx = get_context(0)

    ernie_model, vocab = get_ernie_model(args.ernie_model, args.cased, ctx, args.dropout_prob)

    dataset = ERNIETaggingDataset(text_vocab=vocab, train_path=config.train_path, dev_path=config.dev_path,
                                  test_path=config.test_path, seq_len=config.seq_len, is_cased=config.cased)
    train_data_loader = dataset.get_train_data_loader(config.batch_size)
    dev_data_loader = dataset.get_dev_data_loader(config.batch_size)
    test_data_loader = dataset.get_test_data_loader(config.batch_size)

    net = ERNIETagger(ernie_model, dataset.num_tag_type, config.dropout_prob)
    net.tag_classifier.initialize(init=mx.init.Normal(0.02), ctx=ctx)
    net.hybridize(static_alloc=True)

    #loss
    loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    loss.hybridize(static_alloc=True)









if __name__ == "__main__":
    print("Ernie Ner")
    args = parse_args()
    main(args)
