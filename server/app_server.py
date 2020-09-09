# encoding: utf-8
"""
python server for multi-thread paddle predict
based on flask and twisted
"""
import init_paths
import time
import ctypes
import json
import conf.server_conf

from util.logger_util import logger
from conf.server_conf import config
from util.pivot import Pivot

from flask import Flask
from flask import jsonify
from flask import abort
from flask import request
from flask import make_response
import sys
import logging

# get main version of python, maybe 3 or 2
PYTHON_VERSION = int(sys.version.split('.')[0])
EXECUTOR = 'executor'
PREDICTOR = 'predictor'
app = Flask(__name__)
log = logging.getLogger(__name__)

class ApiResult(object):
    """
    API 返回结果
    """
    orderNo = int
    error_code = int
    errpr_message = str
    data = dict

    def __init__(self, code=0, orderNo=None, message='', data=None):
        self.orderNo = orderNo
        self.error_code = code
        self.error_message = message
        self.data = data

    def success(self, data=None):
        """成功返回值"""
        self.data = data

    def error(self, code=-1, message='error', data=None):
        """失败返回值"""
        self.error_code = code
        self.error_message = message
        self.data = data


@app.route('/infer', methods=['POST'])
def recognize():
    """
    处理post请求,并返回处理后的结果
    """
    if not request.json:
        abort(400)
    if 'orderNo' in request.json:
        orderNo = request.json['orderNo']
        logger.set_logid(str(orderNo))
    else:
        logger.set_auto_logid()
        orderNo = logger.get_logid()
    orderNo = int(orderNo)
    result = ApiResult(orderNo=orderNo)
    start_time = time.time()

    try:
        origin_data = request.json
        log.debug('origin_data: %s' % repr(origin_data))
        task_data = process_data(origin_data)
        log.debug('task_data_orderNo: %s' % repr(task_data))
        start_time = time.time()
        log.debug("prepare start_time cost time: {}".format("%2.2f sec" % start_time))
        max_request_time = float(config.get('max_request_time'))
        index = pivot.set_task_queue(task_data, max_request_time)
        if index is not None:
            text = pivot.get_result_list(index)
            log.debug('text: %s' % repr(text))
            pred = dict()
            pred['result'] = text
            pred['orderNo'] = orderNo
            result.success(data=pred)
        else:
            result.error(message="busy, wait then retry")
    except Exception as err:
        logger.error("infer exception: {}".format(err))
        result.error(message=str(err))

    period = time.time() - start_time
    logger.info("request cost:{}".format("%2.2f sec" % period))
    return json.dumps(result, default=lambda o: o.__dict__)


@app.route('/')
def index():
    """
    访问root
    """
    logger.info("visit root page")
    return 'Index Page'


@app.route('/state')
def state():
    """
    访问state
    """
    logger.info("visit state page")
    return '0'


@app.errorhandler(404)
def not_found(error):
    """
    如果访问出错，返回404错误
    """
    return make_response(jsonify({'error': 'Not found'}), 404)


if __name__ == '__main__':
    """
    web server start
    """

    if len(sys.argv) > 2:
        logger.critical("too less argv, need a start profile, run as: python app_server.py dev")
        exit(1)
    else:
        # conf.server_conf.ServerConfig.init_evn(sys.argv[1])
        conf.server_conf.ServerConfig.init_evn("dev")

        from twisted.internet import reactor
        from twisted.web.resource import Resource
        from twisted.web.server import Site
        from twisted.web.wsgi import WSGIResource
        from util.consumer import InferConsumer
        from business_service import process_data

        pivot = Pivot(int(config.get("max_task_num")))
        infer_consumer = InferConsumer(pivot, float(config.get("max_get_time")))
        port = int(config.get("flask.server.port"))
        work_thread_count = int(config.get('work.thread.count'))
        site_root = str(config.get('server.root.site'))

        reactor.suggestThreadPoolSize(int(work_thread_count))
        flask_site = WSGIResource(reactor, reactor.getThreadPool(), app)
        root = Resource()
        root.putChild(site_root if PYTHON_VERSION == 2 else bytes(site_root, encoding='utf-8'), flask_site)
        logger.info("start app listen port:{} thread pool size:{} site root:{}"
                    .format(port, work_thread_count, site_root))
        reactor.listenTCP(port, Site(root))

        # 把主线程设置为infer_consumer的守护线程，如果主线程结束了，infer_consumer线程就跟着结束
        infer_consumer.setDaemon(True)
        infer_consumer.start()

        reactor.run()

