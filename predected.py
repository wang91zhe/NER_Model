import os
import argparse
import re
import mxnet as mx

from ner_utils import get_context, str2bool, get_ernie_model, dump_metadata
from predected_dataset import ERNIETaggingPredectedDataset, convert_arrays_to_text
from model import ERNIETagger, attach_prediction

def parse_args():
    """Parse command line arguments."""
    arg_parser = argparse.ArgumentParser(
        description='Train a ERNIE-based named entity recognition model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('--save-checkpoint-prefix', type=str, default="./model/chinese_english_good_name/model_007.params",
                            help='Prefix of model checkpoint file')

    # bert options
    arg_parser.add_argument('--ernie-model', type=str, default='ernie_12_768_12',
                            help='Name of the ERNIE model')
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
    arg_parser.add_argument('--num-epochs', type=int, default=10, help='Number of epochs to train')
    arg_parser.add_argument('--optimizer', type=str, default='adam',
                            help='Optimization algorithm to use')
    arg_parser.add_argument('--learning-rate', type=float, default=5e-5,
                            help='Learning rate for optimization')
    arg_parser.add_argument('--warmup-ratio', type=float, default=0.1,
                            help='Warmup ratio for learning rate scheduling')
    args = arg_parser.parse_args()
    return args

def trans_data_style(sentence):
    text, label = "", ""
    count = 0
    words = sentence.strip('\n').split(" ")
    pattern = re.compile("[\\u4e00-\\u9fa5]+")
    for txt in words:
        if pattern.findall(txt):
            for w in txt:
                if count == 0:
                    text = w
                    label = "O"
                    count = 1
                    continue
                text += "\002" + w
                label += "\002" + "O"
            continue
        if count == 0:
            text = txt
            label = "O"
            count = 1
            continue
        text += "\002" + txt
        label += "\002" + "O"
    return text + "\t" + label


def predected(config):
    ctx = get_context(0)
    origin_text = "遗产基因分析仪用底座 Basefor Heritage Genetic Analyzer 26861.37 26861.37"
    sentence = trans_data_style(origin_text)
    tag_list = ["EN-B", "EN-I", "CN-B", "CN-I", "O"]

    ernie_model, vocab = get_ernie_model(args.ernie_model, args.cased, ctx, args.dropout_prob)

    dataset = ERNIETaggingPredectedDataset(text_vocab=vocab, scentences=sentence, seq_len=config.seq_len,
                                           is_cased=config.cased, tag_list=tag_list)
    data_loader = dataset.get_predected_data_loader(config.batch_size)

    net = ERNIETagger(ernie_model, dataset.num_tag_types, config.dropout_prob)
    net.load_params(config.save_checkpoint_prefix, ctx=ctx)

    for batch_id, (text_ids, _, valid_length, tag_ids, _, out) in \
            enumerate(attach_prediction(data_loader, net, ctx, is_train=False)):

        # convert results to numpy arrays for easier access
        np_text_ids = text_ids.astype('int32').asnumpy()
        np_pred_tags = out.argmax(axis=-1).asnumpy()
        np_valid_length = valid_length.astype('int32').asnumpy()
        # np_true_tags = tag_ids.asnumpy()

        predect_text = convert_arrays_to_text(vocab, dataset.tag_vocab, np_text_ids,
                                              np_pred_tags, np_valid_length)
        print(predect_text)

    print("OK")


if __name__ == "__main__":
    print("Model Predected")
    args = parse_args()
    predected(args)
