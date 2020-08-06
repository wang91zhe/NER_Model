import numpy as np
import logging
from collections import namedtuple
import gluonnlp as nlp
import mxnet as mx

logging.getLogger().setLevel(logging.DEBUG)
TaggedToken = namedtuple('TaggedToken', ['text', 'tag'])
NULL_TAG = 'O'

def read_bio(file_path):
    logging.info("read BIO Tagging")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readline()
        sentence_list = []
        for line in lines:
            cur_sentence = []
            text, label = line.strip('\n').split('\t')
            if "text_a" == text and "label" == label:
                continue
            for txt, lab in zip(text, label):
                cur_sentence.append(TaggedToken(text=txt, tag=lab))
            sentence_list.append(cur_sentence)

        return sentence_list

def ernie_tokenize_sentence(sentence, tokenizer):
    ret = []
    for words in sentence:
        sub_token_text = tokenizer(words.text)
        ret.append(TaggedToken(text=sub_token_text[0], tag=words.tag))
        #load after label
        ret += [TaggedToken(text=sub_token, tag=NULL_TAG)
                for sub_token in sub_token_text[1:]]
    return ret

def load_segment(file_path, ernie_tokenizer):
    logging.info("load scentence in %s", file_path)
    sentence_list = read_bio(file_path)
    subword_sentences = [ernie_tokenize_sentence(sentence, ernie_tokenizer)
                         for sentence in sentence_list]
    logging.info('load %s, its max seq len: %d',
                 file_path, max(len(sentence) for sentence in subword_sentences))
    return subword_sentences


class ERNIETaggingDataset:
    def __init__(self, text_vocab, train_path, dev_path, test_path, seq_len, is_cased,
                 tag_vocab=None):
        self.text_vocab = text_vocab
        self.seq_len = seq_len

        self.ernie_tokenizer = nlp.data.BERTTokenizer(self.text_vocab, lower=not is_cased)

        train_sentence = [] if train_path is None else load_segment(train_path, self.ernie_tokenizer)

        all_sentences = train_sentence

        if tag_vocab is None:
            logging.info('Indexing tags...')
            tag_counter = nlp.data.count_tokens([token.tag for sentence in all_sentences for token in sentence])
            self.tag_vocab = nlp.Vocab(tag_counter, padding_token=NULL_TAG,
                                       bos_token=None, eos_token=None, unknown_token=None)
        else:
            self.tag_vocab = tag_vocab

        self.null_tag_index = self.tag_vocab[NULL_TAG]

        self.train_inputs = [self._encode_as_input(sentence) for sentence in train_sentence]

        logging.info('tag_vocab: %s', self.tag_vocab)

    def _encode_as_input(self, sentence):
        # check whether the given sequence can be fit into `seq_len`.
        assert len(sentence) <= self.seq_len - 2, \
            'the number of tokens {} should not be larger than {} - 2. offending sentence: {}' \
                .format(len(sentence), self.seq_len, sentence)

        text_token = [[self.text_vocab.cls_token]  + [token.text for token in sentence] +
                      [self.text_vocab.sep_token]]
        padded_text_ids = (self.text_vocab.to_indices(text_token) +
                           ([self.text_vocab(self.text_vocab.padding_token)]*(self.seq_len - len(text_token))))

        tags = [NULL_TAG] + [token.tag for token in sentence] + [NULL_TAG]
        padded_tag_ids = (self.tag_vocab.to_indices(tags) +
                           ([self.tag_vocab(NULL_TAG)]*(self.seq_len - len(tags))))

        assert len(text_token) == len(tags)
        assert len(padded_text_ids) == len(text_token)
        assert  len(padded_text_ids) == self.seq_len

        valid_length = len(text_token)

        # in sequence tagging problems, only one sentence is given
        token_types = [0] * self.seq_len

        np_tag_ids = np.array(padded_tag_ids, "int32")

        # gluon batchify cannot batchify numpy.bool? :(
        flag_nonnull_tag = (np_tag_ids != self.null_tag_index).astype('int32')

        return (np.array(padded_text_ids, dtype='int32'),
                np.array(token_types, dtype='int32'),
                np.array(valid_length, dtype='int32'),
                np_tag_ids,
                flag_nonnull_tag)

    @staticmethod
    def _get_data_loader(inputs, shuffle, batch_size):
        return mx.gluon.data.DataLoader(inputs, batch_size=batch_size, shuffle=shuffle,
                                        last_batch='keep')

    def get_train_data_loader(self, batch_size):
        return self._get_data_loader(self.train_inputs, shuffle=True, batch_size=batch_size)

    @property
    def num_tag_types(self):
        return len(self.tag_vocab)



