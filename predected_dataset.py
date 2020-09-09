import os
import logging
import numpy as np
from collections import namedtuple
import mxnet as mx
import gluonnlp as nlp

logging.getLogger().setLevel(logging.DEBUG)

TaggedToken = namedtuple('TaggedToken', ['text', 'tag'])
PredictedToken = namedtuple('PredictedToken', ['text', 'true_tag', 'pred_tag'])
NULL_TAG = 'X'

def read_bio(scentence):
    sentence_list = []
    cur_sentence = []
    text, label = scentence.strip('\n').split('\t')
    arr_text = text.split("\002")
    arr_label = label.split("\002")
    for txt, lab in zip(arr_text, arr_label):
        txt = txt.lower()
        cur_sentence.append(TaggedToken(text=txt, tag=lab))
    sentence_list.append(cur_sentence)
    return sentence_list


def ernie_tokenize_sentence(sentence, tokenizer):
    ret = []
    count = 0
    for words in sentence:
        sub_token_text = tokenizer(words.text)
        if sub_token_text is None or len(sub_token_text) == 0:
            continue
        ret.append(TaggedToken(text=sub_token_text[0], tag=words.tag))
        #load after label
        ret += [TaggedToken(text=sub_token, tag=NULL_TAG)
                for sub_token in sub_token_text[1:]]
    return ret

def load_segment(scentences, ernie_tokenizer):
    sentence_list = read_bio(scentences)
    subword_sentences = [ernie_tokenize_sentence(sentence, ernie_tokenizer)
                         for sentence in sentence_list]
    logging.info('its max seq len: %d',
                 max(len(sentence) for sentence in subword_sentences))
    return subword_sentences

class ERNIETaggingPredectedDataset:
    def __init__(self, text_vocab, scentences, seq_len, is_cased, tag_list, tag_vocab=None):
        self.text_vocab = text_vocab
        self.seq_len = seq_len
        self.tag_list = tag_list

        self.ernie_tokenizer = nlp.data.BERTTokenizer(self.text_vocab, lower=not is_cased)

        predected_sentence = [] if scentences is None else load_segment(scentences, self.ernie_tokenizer)

        if tag_vocab is None:
            logging.info('Indexing tags...')
            tag_counter = nlp.data.count_tokens([tag for tag in self.tag_list])
            self.tag_vocab = nlp.Vocab(tag_counter, padding_token=NULL_TAG,
                                       bos_token=None, eos_token=None, unknown_token=None)
        else:
            self.tag_vocab = tag_vocab

        self.null_tag_index = self.tag_vocab[NULL_TAG]

        self.predect_inputs = [self._encode_as_input(sentence) for sentence in predected_sentence]

        logging.info('tag_vocab: %s', self.tag_vocab)

    def _encode_as_input(self, sentences):
        # check whether the given sequence can be fit into `seq_len`.
        sentence = sentences[:self.seq_len - 2]
        assert len(sentence) <= self.seq_len - 2, \
            'the number of tokens {} should not be larger than {} - 2. offending sentence: {}' \
                .format(len(sentence), self.seq_len, sentence)

        text_token = ([self.text_vocab.cls_token]  + [token.text for token in sentence] +
                      [self.text_vocab.sep_token])
        padded_text_ids = (self.text_vocab.to_indices(text_token) +
                           ([self.text_vocab[self.text_vocab.padding_token]]*(self.seq_len - len(text_token))))

        tags = [NULL_TAG] + [token.tag for token in sentence] + [NULL_TAG]
        padded_tag_ids = (self.tag_vocab.to_indices(tags) +
                           ([self.tag_vocab[NULL_TAG]]*(self.seq_len - len(tags))))

        assert len(text_token) == len(tags)
        assert len(padded_text_ids) == len(padded_tag_ids)
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

    def get_predected_data_loader(self, batch_size):
        return self._get_data_loader(self.predect_inputs, shuffle=False, batch_size=batch_size)

    @property
    def num_tag_types(self):
        return len(self.tag_vocab)


def convert_arrays_to_text(text_vocab, tag_vocab,
                           np_text_ids, np_pred_tags, np_valid_length):
    predect_text = ""
    label = ""
    for sample_index in range(np_valid_length.shape[0]):
        sample_len = np_valid_length[sample_index]

        for i in range(1, sample_len - 1):
            token_text = text_vocab.idx_to_token[np_text_ids[sample_index, i]]
            pred_tag = tag_vocab.idx_to_token[int(np_pred_tags[sample_index, i])]
            if i == 1:
                predect_text = token_text
                continue
            if pred_tag.endswith('-B') or pred_tag == 'O':
                predect_text += u"/" + label + "\002" + token_text
                if pred_tag == 'O':
                    label = 'O'
            elif pred_tag.endswith('-I'):
                predect_text += " " + token_text
                label = pred_tag.split('-')[0]

    if label != "":
        predect_text += u"/" + label



    return predect_text