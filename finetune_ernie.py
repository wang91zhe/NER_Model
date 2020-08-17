import os
import numpy as np
import random
import argparse
import mxnet as mx
import logging
import gluonnlp as nlp

from ner_utils import get_context, str2bool, get_ernie_model, dump_metadata
from data import ERNIETaggingDataset, convert_arrays_to_text
from model import ERNIETagger, attach_prediction

# seqeval is a dependency that is specific to named entity recognition.
import seqeval.metrics

logging.getLogger().setLevel(logging.DEBUG)

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

    arg_parser.add_argument('--save-checkpoint-prefix', type=str, default="./model/model",
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
    arg_parser.add_argument('--num-epochs', type=int, default=20, help='Number of epochs to train')
    arg_parser.add_argument('--optimizer', type=str, default='adam',
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

    net = ERNIETagger(ernie_model, dataset.num_tag_types, config.dropout_prob)
    net.tag_classifier.initialize(init=mx.init.Normal(0.02), ctx=ctx)
    net.hybridize(static_alloc=True)

    #loss
    loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    loss.hybridize(static_alloc=True)

    setp_size = config.batch_size
    max_num_train_steps = (int(len(dataset.train_inputs)) * config.num_epochs) / setp_size
    num_wramup_steps = int(max_num_train_steps * config.warmup_ratio)

    #optimizer
    optimizer_params = {"learning_rate": config.learning_rate}
    trainer = mx.gluon.Trainer(net.collect_params(), config.optimizer, optimizer_params)

    # collect differentiable parameters
    logging.info('Collect params...')
    # do not apply weight decay on LayerNorm and bias terms
    for _, v in net.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0
    params = [p for p in net.collect_params().values() if p.grad_req != 'null']

    if config.save_checkpoint_prefix is not None:
        logging.info('dumping metadata...')
        dump_metadata(config, tag_vocab=dataset.tag_vocab)

    def train(data_loader, start_step_num):
        """Training loop."""
        step_num = start_step_num
        # logging.info('current starting step num: %d', step_num)
        for batch_id, (_, _, _, tag_ids, flag_nonnull_tag, out) in \
                enumerate(attach_prediction(data_loader, net, ctx, is_train=True)):
            # logging.info('training on batch index: %d/%d', batch_id, len(data_loader))

            # step size adjustments
            #leearning_rate decay
            step_num += 1
            if step_num < num_wramup_steps:
                new_lr = config.learning_rate * step_num / num_wramup_steps
            else:
                offset = ((step_num - num_wramup_steps) * config.learning_rate /
                          (max_num_train_steps - num_wramup_steps))
                new_lr = config.learning_rate - offset
            trainer.set_learning_rate(new_lr)

            with mx.autograd.record():
                loss_value = loss(out, tag_ids,
                                  flag_nonnull_tag.expand_dims(axis=2)).mean()

            loss_value.backward()
            nlp.utils.clip_grad_global_norm(params, 1)
            trainer.step(1)

            pred_tags = out.argmax(axis=-1)
            # logging.info('loss_value: %6f', loss_value.asscalar())

            num_tag_preds = flag_nonnull_tag.sum().asscalar()
            logging.info(
                'loss_value: %6f, accuracy: %6f', loss_value.asscalar(), (((pred_tags == tag_ids) * flag_nonnull_tag).sum().asscalar()
                                  / num_tag_preds))
        return step_num

    def evaluate(data_loader):
        """Eval loop."""
        predictions = []

        for batch_id, (text_ids, _, valid_length, tag_ids, _, out) in \
                enumerate(attach_prediction(data_loader, net, ctx, is_train=False)):
            # logging.info('evaluating on batch index: %d/%d', batch_id, len(data_loader))

            # convert results to numpy arrays for easier access
            np_text_ids = text_ids.astype('int32').asnumpy()
            np_pred_tags = out.argmax(axis=-1).asnumpy()
            np_valid_length = valid_length.astype('int32').asnumpy()
            np_true_tags = tag_ids.asnumpy()

            predictions += convert_arrays_to_text(vocab, dataset.tag_vocab, np_text_ids,
                                                  np_true_tags, np_pred_tags, np_valid_length)

        all_true_tags = [[entry.true_tag for entry in entries] for entries in predictions]
        all_pred_tags = [[entry.pred_tag for entry in entries] for entries in predictions]
        seqeval_f1 = seqeval.metrics.f1_score(all_true_tags, all_pred_tags)
        return seqeval_f1

    best_dev_f1 = 0.0
    last_test_f1 = 0.0
    best_epoch = -1

    last_epoch_step_num = 0
    for epoch_index in range(config.num_epochs):
        last_epoch_step_num = train(train_data_loader, last_epoch_step_num)
        train_f1 = evaluate(train_data_loader)
        dev_f1 = evaluate(dev_data_loader)
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_epoch = epoch_index
            # logging.info('update the best dev f1 to be: %3f', best_dev_f1)
            test_f1 = evaluate(test_data_loader)
            logging.info('test f1: %3f', test_f1)
            last_test_f1 = test_f1

            # save params
            params_file = config.save_checkpoint_prefix + '_{:03d}.params'.format(epoch_index)
            # logging.info('saving current checkpoint to: %s', params_file)
            net.save_parameters(params_file)

        logging.info('epoch: %d, current_best_epoch: %d, train_f1: %3f, dev_f1: %3f, previous_best_dev f1: %3f',
                     epoch_index, best_epoch, train_f1, dev_f1, best_dev_f1)

        # logging.info('current best epoch: %d', best_epoch)

    logging.info('best epoch: %d, best dev f1: %3f, test f1 at tha epoch: %3f',
                 best_epoch, best_dev_f1, last_test_f1)


if __name__ == "__main__":
    print("Ernie Ner")
    args = parse_args()
    main(args)
