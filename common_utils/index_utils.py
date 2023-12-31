# -*- coding: UTF-8 -*-
from llama_index import Document
from common_utils.milvus_vector_store import CUMilvusVectorStore
from llama_index import VectorStoreIndex
from llama_index import ServiceContext
from llama_index.embeddings.base import BaseEmbedding
from llama_index.llms.langchain import LangChainLLM
from llama_index.llms.base import LLM
from common_utils.remote_textgen_model import CUTextGen
from pymilvus import utility
from pymilvus import MilvusClient
from pymilvus import Collection
from common_utils.node_parser import QANodeParser
import config
from common_utils.md5 import get_md5_value
from common_utils.redis_utils import RedisUtils
import requests, json

def create_milvus_client():
    return MilvusClient(uri=config.MILVUS_URI, token=config.MILVUS_TOKEN)

def create_llm(temperature: float = 0.0, presence_penalty: float = 0.0, frequency_penalty: float = 0.0, is_qa: bool = False) -> LLM:
    return create_llm_for_model(config.MODEL_URL, temperature, presence_penalty, frequency_penalty, is_qa)

def create_llm_for_model(model_url: str, temperature: float = 0.0, presence_penalty: float = 0.0, frequency_penalty: float = 0.0, is_qa: bool = False) -> LLM:
    textgen_llm = CUTextGen(model_url=model_url)
    if is_qa:
        textgen_llm.is_qa = True
    textgen_llm.temperature = temperature
    textgen_llm.presence_penalty = presence_penalty
    textgen_llm.frequency_penalty = frequency_penalty
    textgen_llm.max_new_tokens = config.MAX_NEW_TOKENS
    textgen_llm.typical_p = config.TYPICAL_P
    textgen_llm.do_sample = config.DO_SAMPLE
    return LangChainLLM(
        llm=textgen_llm
    )

def create_vec_store(embed_model: BaseEmbedding, collection_name: str, radius: float = 0.5, is_qa: bool = False):
    text_embedding = embed_model.get_text_embedding('你好')
    # print('text_embedding:', len(text_embedding))
    return CUMilvusVectorStore(is_qa = is_qa, uri=config.MILVUS_URI, dim=len(text_embedding), collection_name=collection_name, radius=radius)

def create_vec_index(llm, embed_model, vector_store):
    # pt = PromptTemplate('Context information is below.\n---------------------\nPlease speak Chinese.\n{context_str}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {query_str}\nAnswer: ')
    # templet = SelectorPromptTemplate(pt)
    service_context = ServiceContext.from_defaults(
        llm=llm, embed_model=embed_model, node_parser=QANodeParser.from_defaults(
            chunk_size=None,
            chunk_overlap=None,
            callback_manager=None,
        ),
    )
    return VectorStoreIndex.from_vector_store(service_context=service_context, vector_store=vector_store)

def create_doc_list(text_list):
    doc_chunks = []
    answer_dict = {}
    for text in text_list:
        # print('text:', text)
        if type(text) == str:
            doc_id = get_md5_value(text)
        else:
            question = text['question']
            answer = text['answer']
            text = question
            doc_id = get_md5_value(text)
            answer_dict['doc_id_' + doc_id] = answer
        doc_chunks.append(create_doc(doc_id, text))
    return (doc_chunks, answer_dict)

def create_doc(doc_id, text):
    return Document(text=text, id_=f'doc_id_{doc_id}')

def append_doc_list(index, doc_list, answer_dict=None):
    print('..........answer_dict len:', len(answer_dict))
    _exist_id_set = exist_id_set(index, doc_list)
    id_list = []
    redis_utils = RedisUtils(config.REDIS_DB)
    for doc_chunk in doc_list:
        if doc_chunk.doc_id not in _exist_id_set:
            # print('....insert:', doc_chunk.doc_id)
            if answer_dict is not None:
                prefix = 'qa-'
                redis_key = prefix + doc_chunk.doc_id
                if isinstance(index, VectorStoreIndex):
                    vector_store = index.vector_store
                    if isinstance(vector_store, CUMilvusVectorStore):
                        redis_key = prefix + vector_store.collection_name + '-' + doc_chunk.doc_id
                # print('.....redis key:', redis_key)
                answer = answer_dict.get(doc_chunk.doc_id)
                if answer is not None:
                    redis_utils.setObjectForever(redis_key, answer, compress=True)
                else:
                    print('.........null question:', doc_chunk.doc_id)
            index.insert(doc_chunk)
        id_list.append(doc_chunk.doc_id)
    return id_list

def append_document(index, doc_chunk):
    if isinstance(index, VectorStoreIndex):
        vector_store = index.vector_store
        if isinstance(vector_store, CUMilvusVectorStore):
            if not vector_store.query_id(doc_chunk.doc_id):
                index.insert(doc_chunk)
                return doc_chunk.doc_id
    return None


def list_id(doc_list):
    id_list = []
    for doc_chunk in doc_list:
        id_list.append(doc_chunk.doc_id)
    return id_list

def exist_id_set(index, doc_list) -> set:
    id_list = list_id(doc_list)
    if isinstance(index, VectorStoreIndex):
        vector_store = index.vector_store
        if isinstance(vector_store, CUMilvusVectorStore):
            return set(vector_store.query_ids(id_list))
    return set()


def delete_collection(collection_name):
    return utility.drop_collection(collection_name)

def drop_vectors(collection_name, data_ids):
    expr = f'id in {str(data_ids)}'
    conn = Collection(collection_name)
    result = conn.delete(expr=expr)
    print('........drop_vectors:', result)
    return result.delete_count == len(data_ids)

def query_id(vector_store: CUMilvusVectorStore, data_id: str):
    return vector_store.query_id(data_id)

# res = conn.query(expr='id in ['doc_id_400687434ccf458d2ad7c14501231fbe']', output_fields=['id'])
# print(res)

def chat(question: str, history_list: list = None, temperature=0.0, presence_penalty=0.0, frequency_penalty=0.0) -> str:
    history = {}
    internal_list = []
    visible_list = []
    if history_list is not None:
        for history_dict in history_list:
            if 'question' in history_dict.keys() and 'answer' in history_dict.keys():
                internal_list.append([str(history_dict['question']), str(history_dict['answer'])])
                visible_list.append([str(history_dict['question']), str(history_dict['answer'])])
    history['internal'] = internal_list
    history['visible'] = visible_list
    # print('.........history:', history)
    param = {
        'max_new_tokens': config.MAX_NEW_TOKENS,
        'do_sample': False,
        'temperature': temperature,
        'presence_penalty': presence_penalty,
        'frequency_penalty': frequency_penalty,
        'top_p': 0.1,
        'typical_p': config.TYPICAL_P,
        'epsilon_cutoff': 0,
        'eta_cutoff': 0,
        'repetition_penalty': 1.18,
        'top_k': 40,
        'min_length': 0,
        'no_repeat_ngram_size': 0,
        'num_beams': 1,
        'penalty_alpha': 0,
        'length_penalty': 1,
        'early_stopping': False,
        'seed': -1,
        'add_bos_token': True,
        'truncation_length': 2048,
        'ban_eos_token': False,
        'skip_special_tokens': True,
        'stopping_strings': [],
        'user_input': question,
        'history': history
    }
    # print('.........param:', param)
    header = {'Content-Type': 'application/json'}
    json_data = json.dumps(param)
    response = requests.post(config.MODEL_URL + '/api/v1/chat', data=json_data, headers=header)
    result = response.content.decode('utf-8')
    # print('........result:', result)
    if result.startswith('{'):
        data_dict = json.loads(result)
        if 'results' in data_dict.keys():
            results = data_dict['results']
            if len(results) > 0:
                history_dict = results[0]
                # print(history_dict)
                if 'history' in history_dict.keys():
                    history_dict = history_dict['history']
                    if 'internal' in history_dict.keys():
                        internal_list = history_dict['internal']
                        if len(internal_list) > 0:
                            data_list = internal_list[len(internal_list) - 1]
                            if len(data_list) > 0:
                                return str(data_list[len(data_list) - 1]).split("\n")[0]
    return ''