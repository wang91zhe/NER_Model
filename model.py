import mxnet as mx
import gluonnlp as nlp
from mxnet.gluon import Block, nn
from contextlib import ExitStack

class ERNIETagger(Block):
    def __init__(self, ernie_model, num_tag_types, dropout_prob, prefix=None, params=None):
        super(ERNIETagger, self).__init__(prefix=prefix, params=params)
        self.ernie_model = ernie_model
        with self.name_scope():
            self.tag_classifier = nn.Dense(units=num_tag_types, flatten=False)
            self.drop_out = nn.Dropout(rate=dropout_prob)

    def forward(self, token_ids, token_types, valid_length):
        ernie_encode = self.ernie_model(token_ids, token_types, valid_length)
        hidden = self.drop_out(ernie_encode)
        out_put = self.tag_classifier(hidden)
        return out_put

def attach_prediction(data_loader, net, ctx, is_train):
    for data in data_loader:
        text_ids, token_types, valid_length, tag_ids, flag_nonnull_tag = \
            [x.astype('float32').as_in_context(ctx) for x in data]

        with ExitStack() as stack:
            if is_train:
                stack.enter_context(mx.autograd.record())
            out = net(text_ids, token_types, valid_length)
        yield text_ids, token_types, valid_length, tag_ids, flag_nonnull_tag, out