# -*- coding: utf-8 -*-
"""
业务相关的写在这里
"""
import numpy as np
import time
import sys
import os
import logging
sys.path.append("..")
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../..")))

log = logging.getLogger(__name__)
from util.logger_util import logger
import util.data_util as data_util
from infer_service import infer_with_executor

PYTHON_VERSION = int(sys.version.split('.')[0])


def business_process(inputs):
    """
    不同业务，数据处理的方式可能不一样，统一在这里修改和控制
    为了充分利用多线程并行的方式来处理数据，所以首先由生产者调用process_data来处理数据，并提交至任务队列
    此处直接从任务队列中获取处理好的数据即可，因此该部分应该和process_data一起修改
    :param inputs:      从任务队列中获取的预处理后的数据
    :return:
    """

    # 数据
    if len(inputs) < 0:
        logger.critical("must set a data")
        exit(1)

    start_time = time.time()
    # 通过不同的model名称来调用模型级联任务中 不同阶段需要用到的模型
    text = ""
    try:
        result = infer_with_executor(inputs)
        period = time.time() - start_time
        if len(result) == 0 or result is None:
            text = ""
        elif len(result) > 0:
            return result
        logger.info(
            "executor infer num {} cost time: {}".format(str(10), "%2.2f sec" % period))
    except Exception:
        log.info("模型预测出现异常".format(Exception))
    return text


def process_data(inputs):
    """
    根据infer_type来处理数据，并返回, 可根据业务需要自行修改
    :param inputs:    原始的输入数据
    :return:
    """
    # 处理数据
    start_time = time.time()
    origin_data = inputs["text"]
    model_name = inputs["model_name"]
    if origin_data is None or len(origin_data) == 0:
        return ""
    data = data_util.read_data(origin_data)
    log.debug('data: %s' % repr(data))
    return [model_name, data]

if __name__ == "__main__":
    business_process("")
