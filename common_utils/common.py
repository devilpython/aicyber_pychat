import os
import datetime
import logging
from logging import getLogger, INFO
from concurrent_log_handler import ConcurrentRotatingFileHandler
from multiprocessing import cpu_count

def initPath(path):
    have_dir = False
    if os.path.exists(path) and not os.path.isfile(path):
        have_dir = True
    if not have_dir:
        os.makedirs(path)

def removeLogging(path):
    LOG_FILE = datetime.datetime.now().strftime('%Y-%m-%d') + '.log'
    log_path = os.path.join(path, LOG_FILE)
    print('remove log file:', log_path)
    try:
        os.remove(log_path)
    except:
        pass

def initLogging(path, log_name):
    print('log initializing...')
    # 日志路径
    log_path = path + '/logs'
    initPath(log_path)
    LOG_FILE = datetime.datetime.now().strftime('%Y-%m-%d') + '.log'
    log_path = os.path.join(log_path, LOG_FILE)
    print('initialize log file:', log_path)

    log = getLogger(log_name)
    rotateHandler = ConcurrentRotatingFileHandler(log_path, "a", 1024 * 1024 * 100, cpu_count())

    formatter = logging.Formatter('%(asctime)s|%(levelname)s|%(message)s')
    rotateHandler.setFormatter(formatter)
    log.addHandler(rotateHandler)
    log.setLevel(INFO)
    print('log initialized...')

def to_float(value, default_value) -> float:
    if value is None:
        return default_value
    try:
        fv = float(str(value))
        return fv
    except:
        return default_value

if __name__ == '__main__':
    LOG_FILE = datetime.datetime.now().strftime('%Y-%m-%d') + '.log'
    current_path = os.path.dirname(os.path.abspath(__file__))
    initLogging(current_path, 'aq_pychat')

    log = getLogger('qa_pychat')
    log.info("Here is a very exciting log message, just for you")