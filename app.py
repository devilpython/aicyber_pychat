from flask import request, jsonify, Flask
from flask_cors import *
from gevent import pywsgi
from common_utils.flask_uploads import UploadSet, configure_uploads, DOCUMENTS
from pymilvus import utility, connections

import config
from common_utils.RequestParameter import getParameter

from logging import getLogger
from common_utils import index_utils, common, text_utils
import os

from llama_index.embeddings import HuggingFaceEmbedding
import json
import requests

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ['MILVUS_CONN_TIMEOUT'] = config.MILVUS_CONN_TIMEOUT

connections.connect(uri=config.MILVUS_URI)

app = Flask(__name__)
CORS(app, supports_credentials=True)

current_path = os.path.dirname(os.path.abspath(__file__))
print('.......current_path:', current_path)
common.initLogging(current_path, 'aicyber_pychat')

files = UploadSet('text', DOCUMENTS)
app.config['UPLOADS_DEFAULT_DEST'] = './upload_files'
configure_uploads(app, files)

log = getLogger('aicyber_pychat')

embed_model = HuggingFaceEmbedding(model_name="./BAAI_bge-base-zh-v1.5")

@app.route('/', methods=['GET', 'POST'])
def getRoot():
    return jsonify({'successful': True, 'message': 'ok', 'version': 'qa pychat 2024.1.2.1'})

@app.route('/favicon.ico', methods=['GET', 'POST'])
def getFavicon():
    return jsonify({'successful': True, 'message': 'favicon.ico'})

@app.route('/upload-texts.do', methods=['GET', 'POST'])
def upload_file():
    collection = getParameter(request, 'collection')
    print('..........collection:', collection)
    if collection is None:
        return jsonify({'successful': False, 'message': u'collection不能为空'})
    if type(collection) is not str:
        return jsonify({'successful': False, 'message': u'collection必须是字符串类型'})

    collection = collection.strip()
    if len(collection) == 0:
        return jsonify({'successful': False, 'message': u'collection不能为空字符串'})

    append = getParameter(request, 'append')
    if append is None:
        append = False
    append = str(append)
    if append.lower() == 'false':
        append = False
    else:
        append = True

    file_name = files.save(request.files['file'])
    file_path = app.config['UPLOADS_DEFAULT_DEST'] + '/text/' + file_name
    # log.info('upload_file......running')

    if not append:
        try:
            utility.drop_collection(collection)
        except Exception as e:
            print('......e:', e)
    try:
        vec_store = index_utils.create_vec_store(embed_model, collection)
        text_list = text_utils.load_text_list(file_path)
        doc_list, answer_dict = index_utils.create_doc_list(text_list)
        # print('.................append docs length:', len(doc_list))
        langchain_llm = index_utils.create_llm(False)
        index = index_utils.create_vec_index(langchain_llm, embed_model, vec_store)
        # print('.................index:', index)
        id_list = index_utils.append_doc_list(index, doc_list, answer_dict)
        # print('.................id_list:', id_list)
        return jsonify({'successful': True, 'message': 'ok', 'ids': id_list})
    except Exception as e:
        print('error:', e)
        log.error('upload_file......exception: ' + str(e))
    return jsonify({'successful': False, 'message': u'api error'})

@app.route('/append-texts.do', methods=['GET', 'POST'])
def append_texts():
    collection = getParameter(request, 'collection')
    if collection is None:
        return jsonify({'successful': False, 'message': u'collection不能为空'})
    if type(collection) is not str:
        return jsonify({'successful': False, 'message': u'collection必须是字符串类型'})
    collection = collection.strip()
    if len(collection) == 0:
        return jsonify({'successful': False, 'message': u'collection不能为空字符串'})

    text_list = getParameter(request, 'text_list')
    if text_list is None:
        return jsonify({'successful': False, 'message': u'text_list不能为空'})
    if type(text_list) is not list:
        return jsonify({'successful': False, 'message': u'text_list必须是字符串列表'})
    # message = text.strip()
    # if len(message) == 0:
    #     return jsonify({'successful': False, 'message': u'text不能为空字符串'})

    try:
        vec_store = index_utils.create_vec_store(embed_model, collection)
        # text_list = text_utils.split_text(text)
        doc_list, answer_dict = index_utils.create_doc_list(text_list)
        langchain_llm = index_utils.create_llm(False)
        index = index_utils.create_vec_index(langchain_llm, embed_model, vec_store)
        id_list = index_utils.append_doc_list(index, doc_list)
        return jsonify({'successful': True, 'message': 'ok', 'ids': id_list})
    except Exception as e:
        print('error:', e)
        log.error('append_texts......exception: ' + str(e))
    return jsonify({'successful': False, 'message': u'api error'})

@app.route('/append-qa.do', methods=['GET', 'POST'])
def append_qa():
    collection = getParameter(request, 'collection')
    if collection is None:
        return jsonify({'successful': False, 'message': u'collection不能为空'})
    if type(collection) is not str:
        return jsonify({'successful': False, 'message': u'collection必须是字符串类型'})
    collection = collection.strip()
    if len(collection) == 0:
        return jsonify({'successful': False, 'message': u'collection不能为空字符串'})

    question = getParameter(request, 'question')
    if question is None:
        return jsonify({'successful': False, 'message': u'question不能为空'})
    if type(question) is not str:
        return jsonify({'successful': False, 'message': u'question必须是字符串类型'})
    question = question.strip()
    if len(question) == 0:
        return jsonify({'successful': False, 'message': u'question不能为空字符串'})

    answer = getParameter(request, 'answer')
    if answer is None:
        return jsonify({'successful': False, 'message': u'answer不能为空'})
    if type(answer) is not str:
        return jsonify({'successful': False, 'message': u'answer必须是字符串类型'})
    answer = answer.strip()
    if len(answer) == 0:
        return jsonify({'successful': False, 'message': u'answer不能为空字符串'})

    try:
        vec_store = index_utils.create_vec_store(embed_model, collection)
        doc_list, answer_dict = index_utils.create_doc_list([text_utils.create_qa_text(question, answer)])
        langchain_llm = index_utils.create_llm(True)
        index = index_utils.create_vec_index(langchain_llm, embed_model, vec_store)
        id_list = index_utils.append_doc_list(index, doc_list, answer_dict)
        if len(id_list) == 1:
            return jsonify({'successful': True, 'message': 'ok', 'id': id_list[0]})
        else:
            return jsonify({'successful': False, 'message': 'qa数据已经存在', 'id': None})
    except Exception as e:
        print('error:', e)
        log.error('append_texts......exception: ' + str(e))
    return jsonify({'successful': False, 'message': u'api error'})

@app.route('/drop-collection.do', methods=['GET', 'POST'])
def drop_collection():
    collection = getParameter(request, 'collection')
    if collection is None:
        return jsonify({'successful': False, 'message': u'collection不能为空'})
    if type(collection) is not str:
        return jsonify({'successful': False, 'message': u'collection必须是字符串类型'})
    collection = collection.strip()
    if len(collection) == 0:
        return jsonify({'successful': False, 'message': u'collection不能为空字符串'})
    try:
        try:
            utility.drop_collection(collection)
        except Exception as e1:
            return jsonify({'successful': False, 'message': str(e1)})
        return jsonify({'successful': True, 'message': 'ok'})
    except Exception as e:
        print('error:', e)
        log.error('drop_collection......exception: ' + str(e))
    return jsonify({'successful': False, 'message': u'api error'})

@app.route('/drop-vectors.do', methods=['GET', 'POST'])
def drop_vectors():
    collection = getParameter(request, 'collection')
    print('..........collection:', collection)
    if collection is None:
        return jsonify({'successful': False, 'message': u'collection不能为空'})
    if type(collection) is not str:
        return jsonify({'successful': False, 'message': u'collection必须是字符串类型'})
    collection = collection.strip()
    if len(collection) == 0:
        return jsonify({'successful': False, 'message': u'collection不能为空字符串'})

    ids = getParameter(request, 'ids')
    if ids is None:
        return jsonify({'successful': False, 'message': u'ids不能为空'})
    if type(ids) is not list:
        return jsonify({'successful': False, 'message': u'ids必须是数字列表'})
    try:
        if index_utils.drop_vectors(collection, ids):
            return jsonify({'successful': True, 'message': 'ok'})
        else:
            return jsonify({'successful': False, 'message': '删除异常'})
    except Exception as e:
        print('error:', e)
        log.error('drop_vectors......exception: ' + str(e))
    return jsonify({'successful': False, 'message': u'api error'})

@app.route('/qa.do', methods=['GET', 'POST'])
def qa():
    collection = getParameter(request, 'collection')
    if collection is None:
        return jsonify({'successful': False, 'message': u'collection不能为空'})
    if type(collection) is not str:
        return jsonify({'successful': False, 'message': u'collection必须是字符串类型'})

    question = getParameter(request, 'question')
    if question is None:
        return jsonify({'successful': False, 'message': u'question不能为空'})
    if type(question) is not str:
        return jsonify({'successful': False, 'message': u'question必须是字符串类型'})
    question = question.strip()
    if len(question) == 0:
        return jsonify({'successful': False, 'message': u'question不能为空字符串'})

    radius = getParameter(request, 'radius')
    temperature = getParameter(request, 'temperature')
    presence_penalty = getParameter(request, 'presence_penalty')
    frequency_penalty = getParameter(request, 'frequency_penalty')

    radius = common.to_float(radius, 0.5)
    temperature = common.to_float(temperature, config.TEMPERATURE)
    presence_penalty = common.to_float(presence_penalty, 0.0)
    frequency_penalty = common.to_float(frequency_penalty, 0.0)
    # print('..........param:', radius, temperature, presence_penalty, frequency_penalty)
    try:
        print('..........qa...question:', question)
        vec_store = index_utils.create_vec_store(embed_model, collection, radius, is_qa=True)
        langchain_llm = index_utils.create_llm(temperature, presence_penalty, frequency_penalty, is_qa=True)
        index = index_utils.create_vec_index(langchain_llm, embed_model, vec_store)
        query_engine = index.as_query_engine()
        answer = query_engine.query(question)
        answer = str(answer)
        if answer.startswith('{'):
            answer_json = json.loads(str(answer))
            print('..........qa...answer:', answer_json['result'])
            return jsonify({'successful': True, 'message': 'ok', 'answer': answer_json['result'], 'prompt': answer_json['prompt'],
                            'model_name': config.MODEL_NAME})
        else:
            return jsonify({'successful': False, 'message': '未查询到数据'})

    except Exception as e:
        print('qa......error:', e)
        log.error('qa......exception: ' + str(e))
    return jsonify({'successful': False, 'message': 'api error'})

@app.route('/search-text.do', methods=['GET', 'POST'])
def search_text():
    collection = getParameter(request, 'collection')
    if collection is None:
        return jsonify({'successful': False, 'message': u'collection不能为空'})
    if type(collection) is not str:
        return jsonify({'successful': False, 'message': u'collection必须是字符串类型'})

    question = getParameter(request, 'question')
    if question is None:
        return jsonify({'successful': False, 'message': u'question不能为空'})
    if type(question) is not str:
        return jsonify({'successful': False, 'message': u'question必须是字符串类型'})
    question = question.strip()
    if len(question) == 0:
        return jsonify({'successful': False, 'message': u'question不能为空字符串'})

    radius = getParameter(request, 'radius')
    temperature = getParameter(request, 'temperature')
    presence_penalty = getParameter(request, 'presence_penalty')
    frequency_penalty = getParameter(request, 'frequency_penalty')

    radius = common.to_float(radius, 0.5)
    temperature = common.to_float(temperature, config.TEMPERATURE)
    presence_penalty = common.to_float(presence_penalty, 0.0)
    frequency_penalty = common.to_float(frequency_penalty, 0.0)
    # print('..........param:', radius, temperature, presence_penalty, frequency_penalty)
    try:
        print('..........search_text...question:', question)
        vec_store = index_utils.create_vec_store(embed_model, collection, radius)
        langchain_llm = index_utils.create_llm(temperature, presence_penalty, frequency_penalty)
        index = index_utils.create_vec_index(langchain_llm, embed_model, vec_store)
        query_engine = index.as_query_engine()
        answer = query_engine.query(question)
        answer = str(answer)
        if answer.startswith('{'):
            answer_json = json.loads(str(answer))
            print('..........search_text...answer:', answer_json['result'])
            return jsonify({'successful': True, 'message': 'ok', 'answer': answer_json['result'],
                            'prompt': answer_json['prompt'],
                            'model_name': config.MODEL_NAME})
        else:
            return jsonify({'successful': False, 'message': '未查询到数据'})

    except Exception as e:
        print('search_text......error:', e)
        log.error('search_text......exception: ' + str(e))
    return jsonify({'successful': False, 'message': 'api error'})

@app.route('/chat.do', methods=['GET', 'POST'])
def chat():
    user_input = getParameter(request, 'user_input')
    if user_input is None:
        return jsonify({'successful': False, 'message': u'user_input不能为空'})
    if type(user_input) is not str:
        return jsonify({'successful': False, 'message': u'user_input必须是字符串类型'})
    user_input = user_input.strip()
    if len(user_input) == 0:
        return jsonify({'successful': False, 'message': u'user_input不能为空字符串'})

    history = getParameter(request, 'history')
    if history is not None and type(history) is not list:
        return jsonify({'successful': False, 'message': u'history必须是列表'})

    temperature = getParameter(request, 'temperature')
    presence_penalty = getParameter(request, 'presence_penalty')
    frequency_penalty = getParameter(request, 'frequency_penalty')

    temperature = common.to_float(temperature, config.TEMPERATURE)
    presence_penalty = common.to_float(presence_penalty, 0.0)
    frequency_penalty = common.to_float(frequency_penalty, 0.0)
    # print('..........param:', radius, temperature, presence_penalty, frequency_penalty)
    try:
        print('..........user_input:', user_input)
        output = index_utils.chat(user_input, history, temperature, presence_penalty, frequency_penalty)
        return jsonify({'successful': True, 'message': 'ok', 'output': output})

    except Exception as e:
        print('chat......error:', e)
        log.error('chat......exception: ' + str(e))
    return jsonify({'successful': False, 'message': 'api error'})

@app.route('/api/v1/generate', methods=['GET', 'POST'])
def generate_api():
    header = {"Content-Type": "application/json"}
    json_data = json.dumps(request.json)
    print('json data:')
    print(json_data)
    response = requests.post(config.MODEL_URL + "/api/v1/generate", data=json_data, headers=header)
    print('response:')
    print(response.content)
    # status_code = response.status_code
    # headers = response.headers
    return response.content

@app.route('/api/v1/chat', methods=['GET', 'POST'])
def chat_api():
    header = {"Content-Type": "application/json"}
    json_data = json.dumps(request.json)
    print('json data:')
    print(json_data)
    response = requests.post(config.MODEL_URL + "/api/v1/chat", data=json_data, headers=header)
    print('response:')
    print(response.content)
    # status_code = response.status_code
    # headers = response.headers
    return response.content

@app.before_request
def write_log_before():
    pass
    # url = request.url
    # if not 'favicon.ico' in url and not 'list.do' in url and not 'searchIndex.do' in url:
    #     log_dict = {}
    #     log_dict['token'] = getParameter(request, 'token')
    #     log_dict['url'] = url
    #     try:
    #         if request.json:
    #             log_dict['post_content'] = request.json
    #         logging.info('request before................%s'%json.dumps(log_dict).decode('unicode_escape'))
    #     except:
    #         pass


@app.after_request
def write_log_after(response):
    url = request.path
    if not 'favicon.ico' in url:
        try:
            request_json = ''
            if request.json:
                request_json = request.json
            log.info('url: ' + url + ", request: " + request_json + ', response: ' + response.get_data())
        except:
            pass
    return response

if __name__ == '__main__':
    pywsgi.WSGIServer(('0.0.0.0', 9090), app).serve_forever()
