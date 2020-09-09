# -*- coding: utf-8 -*-
"""
infer service, just an example
"""
import sys
import os
sys.path.append("..")
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
import logging
import argparse
logging.getLogger().setLevel(logging.DEBUG)

PYTHON_VERSION = int(sys.version.split('.')[0])
EXECUTOR = 'executor'
PREDICTOR = 'predictor'

from conf.server_conf import config
from predict.executor_manager import executor_manager
from ner_utils import get_context, str2bool
from predected_dataset import ERNIETaggingPredectedDataset, convert_arrays_to_text
from model import ERNIETagger, attach_prediction

def infer_with_executor(inputs):
    """
    :param model_name: 需要进行infer的模型名字
    :param infer_data:
    :param *args:
    :return:
    """
    try:
        model_name = inputs[0]
        sentence = inputs[-1]
        ctx = executor_manager.get_ctx("gpu")
        logging.debug('infer model name: %s' % repr(model_name))
        args = config_args()
        net = executor_manager.get_model_stuff(model_name)
        vocab = executor_manager.get_vocab("vocab_name")
        tag_map = executor_manager.get_label_map(model_name)
        tag_list = [tag for tag in tag_map.keys()]

        dataset = ERNIETaggingPredectedDataset(text_vocab=vocab, scentences=sentence, seq_len=args.seq_len,
                                               is_cased=args.cased, tag_list=tag_list)
        data_loader = dataset.get_predected_data_loader(args.batch_size)

        for batch_id, (text_ids, _, valid_length, tag_ids, _, out) in \
                enumerate(attach_prediction(data_loader, net, ctx, is_train=False)):

            # convert results to numpy arrays for easier access
            np_text_ids = text_ids.astype('int32').asnumpy()
            np_pred_tags = out.argmax(axis=-1).asnumpy()
            np_valid_length = valid_length.astype('int32').asnumpy()
            # np_true_tags = tag_ids.asnumpy()

            predect_text = convert_arrays_to_text(vocab, dataset.tag_vocab, np_text_ids,
                                                  np_pred_tags, np_valid_length)
            words = predect_text.split("\002")
            new_text = ""
            for word in words:
                if "##" in word:
                    tem_word = word.split(" ")
                    tem_text = ""
                    for w in tem_word:
                        if "##" in w:
                            txt = w.replace("##", "")
                            tem_text = tem_text.rstrip() + txt + " "
                        else:
                            tem_text += w + " "
                    new_text += tem_text + "\002"
                else:
                    new_text += word + "\002"

            return new_text
    except Exception:
        logging.info("模型inputs data出现异常".format(Exception))

    return ""

def config_args():
    arg_parser = argparse.ArgumentParser(
        description='Train a ERNIE-based named entity recognition model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # bert options
    arg_parser.add_argument('--ernie-model', type=str, default='bert_12_768_12',
                            help='Name of the ERNIE model')
    # optimization parameters
    arg_parser.add_argument('--seq-len', type=int, default=256,
                            help='The length of the sequence input to BERT.'
                                 ' An exception will raised if this is not large enough.')
    arg_parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    arg_parser.add_argument('--cased', type=str2bool, default=False,
                            help='Path to the development data file')
    args = arg_parser.parse_args()
    return args