MILVUS_URI = 'http://localhost:19530'
MILVUS_CONN_TIMEOUT = '5000'
MILVUS_TOKEN = ''

MODEL_URL = 'http://127.0.0.1:5000'
MAX_NEW_TOKENS = 1000
TEMPERATURE = 0.5
TYPICAL_P = 0.1
DO_SAMPLE = False

MODEL_NAME = 'Atom-7B-Chat'

# redis
REDIS_TYPE = 'single'  # single为单机模式,sentinel为哨兵模式
REDIS_SENTINEL_MASTER_NAME = 'mymaster'  # 哨兵集群的主服务器名字
REDIS_SERVER = '127.0.0.1'
REDIS_PASSWORD = 'MyAicyber201415926!'
REDIS_PORT = 6379
REDIS_DB = 0
