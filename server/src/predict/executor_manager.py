# -*- coding: utf-8 -*-
"""
manage executor
"""
import os
import json
import sys
import logging
import paddle.fluid.dygraph as FD

sys.path.append("..")
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../..")))

from conf.server_conf import config
from ner_utils import get_context, get_ernie_model
from model import ERNIETagger

log = logging.getLogger(__name__)

# get main version of python, maybe 3 or 2
import sys
PYTHON_VERSION = int(sys.version.split('.')[0])
PRIVATE = 'private'


class ExecutorManager(object):
    """
    executor manager, hold a multi-thread safe queue
    worker threads compete to get executor
    """
    CPU_DEVICE = 'cpu'
    GPU_DEVICE = 'gpu'

    def __init__(self):
        """
        create executor manager
        """
        self.get_executor_timeout = float(config.get('get.executor.timeout', default_value=0.5))
        model_dir = config.get('model.dir')
        # 将地址的json字符串转为字典，key为模型的名称，value为模型的地址，放便级联使用
        label_path = config.get('label_map')

        # 模型的字典
        self.model_dict = {}
        self.label_map = {}
        self.vocab_dict = {}
        self.ctx = {}
        self._load_label_map(label_path)
        #加载模型
        with open(model_dir, 'r', encoding='utf-8') as f:
            model_dir_data = json.load(f)
            for model_name in model_dir_data.keys():
                num_label = len(self.label_map[model_name])
                self._load_model(model_dir_data[model_name], model_name, num_label)

    def get_model_stuff(self, model_name):
        """
        根据需要的模型名称来获取infer的信息，这里的模型名称对应config中的名称
        infer的信息，包括exe, startup_prog
        """
        model = self.model_dict[model_name]
        return model

    def _load_model(self, model_path, model_name, num_label):
        """
        加载模型
        """
        gpu = config.get('gpu')
        ctx = get_context(int(gpu))
        pretrain_model = config.get('from_pretrained')
        cased = config.get('cased')
        is_cased = False
        if cased == "True":
            is_cased = True
        dropout_prob = float(config.get('dropout_prob'))
        ernie_model, vocab = get_ernie_model(pretrain_model, is_cased, ctx, dropout_prob)
        model = ERNIETagger(ernie_model, num_label + 1, dropout_prob)
        model.load_params(model_path, ctx=ctx)

        self.model_dict[model_name] = model
        self.vocab_dict["vocab_name"] = vocab
        self.ctx["gpu"] = ctx

    def get_vocab(self, vocab_name):
        return self.vocab_dict[vocab_name]

    def get_ctx(self, ctx_name):
        return self.ctx[ctx_name]

    def get_label_map(self, model_name):
        return self.label_map[model_name]

    def _load_label_map(self, label_path):
        with open(label_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        for model_name in json_data.keys():
            label_dict = json_data[model_name]
            self.label_map[model_name] = label_dict


executor_manager = ExecutorManager()
