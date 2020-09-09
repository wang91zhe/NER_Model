# -*- coding: utf-8 -*-
import sys
PYTHON_VERSION = int(sys.version.split('.')[0])
import requests
import threading
import time


url = "http://127.0.0.1:8081/predictor/infer"

class TestThread(threading.Thread):
    def __init__(self, thread_id, loop_count):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.loop_count = loop_count

    def run(self):

        params = {
            'orderNo': "39269090",
            'text': "汽车遮脏垫无品牌，不享惠，赠品，汽车保养修理工防止弄脏衣服用，PU+涤纶，无品牌，无型号"
        }

        print("thread id:{} start loop".format(self.thread_id))
        for i in range(self.loop_count):
            time_start = time.time()
            response = requests.post(url, json=params)
            total_cost = (time.time() - time_start) * 1000
            print('thread id:{} request cost: {}ms\n'.format(self.thread_id, total_cost))

        print("thread id:{} loop {} times end".format(self.thread_id, self.loop_count))


if __name__ == '__main__':
    loop_count = 10
    thread_count = 5
    thread_list = []
    for i in range(thread_count):
        thread = TestThread(i, loop_count)
        thread_list.append(thread)
    start_time = time.time()
    for i in range(len(thread_list)):
        thread_list[i].start()
    for i in range(len(thread_list)):
        thread_list[i].join()
    cost_time = time.time() - start_time
    print("end test, total cost: {}".format(cost_time))
