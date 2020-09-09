# -*- coding: utf-8 -*-
"""
数据处理的方式写在这里
"""
import re
import sys

PYTHON_VERSION = int(sys.version.split('.')[0])

def read_data(inputs):
    text = trans_data_style(inputs)
    return text

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




