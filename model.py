import mxnet as mx
import gluonnlp as nlp
from mxnet.gluon import Block, nn

class ERNIETagger(Block):
    def __init__(self, ernie_model, num_tag_types, dropout_prob, prefix=None, params=None):
        super(self, ERNIETagger).__init__(prefix=prefix, params=params)
        self.ernie_model = ernie_model
        with self.name_scope():
            self.tag_classifier = nn.Dense(units=num_tag_types)
            self.drop_out = nn.Dropout(rate=dropout_prob)

    def forward(self, token_ids, token_types, valid_length):
        ernie_encode = self.ernie_model(token_ids, token_types, valid_length)
        hidden = self.drop_out(ernie_encode)
        out_put = self.tag_classifier(hidden)
        return out_put
