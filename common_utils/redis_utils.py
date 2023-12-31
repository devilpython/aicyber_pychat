# -*- coding: UTF-8 -*-
import struct
import config
import redis
import pickle
import datetime
import zlib
import logging
import time
from redis.sentinel import Sentinel

NoCachePrefix = ['vec-', 'hanlp-', 'bert-']
OutOfRedis = ['flow-key-for-']
RedisCache = {}

class RedisUtils:
    __pool_dict = {}
    __sentinel = None

    def __init__(self, db=0):
        if (not hasattr(config, 'REDIS_TYPE') or config.REDIS_TYPE == 'single') and db not in RedisUtils.__pool_dict.keys():
            logging.info('init redis.......db=%d' % db)
            try:
                if hasattr(config, 'REDIS_PASSWORD'):
                    RedisUtils.__pool_dict[db] = redis.ConnectionPool(host=config.REDIS_SERVER, password=config.REDIS_PASSWORD, port=config.REDIS_PORT, db=db)
                else:
                    RedisUtils.__pool_dict[db] = redis.ConnectionPool(host=config.REDIS_SERVER, port=config.REDIS_PORT, db=db)
            except Exception as e:
                logging.error('RedisUtils.__init_(db=%d) message: %s' % (db, e.message))
        elif config.REDIS_TYPE == 'sentinel':
            self.__sentinel = Sentinel(config.REDIS_SENTINEL_LIST)
        self.__db = db

    def setRedisCache(self, redis_cache):
        if hasattr(config, 'LOCAL_CACHE') and config.LOCAL_CACHE:
            global RedisCache
            RedisCache = redis_cache

    def getDB(self):
        return self.__db

    # 设置对象
    def setObject(self, key, obj, time_limit=None, compress=True):
        data_str = pickle.dumps(obj)
        if compress:
            data_str = zlib.compress(data_str)
        return self.set(key, data_str, time_limit)

    # 设置永久对象
    def setObjectForever(self, key, obj, compress=True):
        data_str = pickle.dumps(obj)
        if compress:
            data_str = zlib.compress(data_str)
        return self.setForever(key, data_str)

    # 获得对象
    def getObject(self, key, compress=True):
        try:
            import time
            # start = time.time()
            data_str = self.get(key)
            # print '.........data_str:', data_str
            # logging.info('...............redis get object key: %s time: %f' % (key, time.time() - start))
            if data_str is not None:
                if compress:
                    # start = time.time()
                    data_str = zlib.decompress(data_str)
                    # logging.info('...............redis decompress object key: %s time: %f'%(key, time.time() - start))
                data = pickle.loads(data_str)
                return data
        except Exception as e:
            logging.error('RedisUtils.getObject(key=%s, db=%d):' % (str(key), self.__db))
            logging.error(e)
        return None

    # 删除对象
    def deleteObject(self, key):
        return self.delete(key)

    def deleteKey(self, key):
        return self.delete(key)

    def getFloatArray(self, key):
        result = self.get(key)
        v = None
        if result is not None:
            if len(result) > 50 and len(result) % 4 == 0:
                v = struct.unpack('%df' % (len(result) / 4), result)
        return v

    def setFloatArray(self, key, value, time_limit=None):
        if value is not None:
            result = struct.pack('%df' % len(value), *value)
            self.set(key, result, time_limit)

    def setFloatArrayForever(self, key, value):
        if value is not None:
            result = struct.pack('%df' % len(value), *value)
            self.setForever(key, result)

    def scan(self, cursor, pattern=None, count=None, db=0):
        client = self.getClient(db)
        if count is None or count <= 0:
            kvps = client.scan(cursor, pattern, None)
        else:
            kvps = client.scan(cursor, pattern, count)
        return kvps

    def hdel(self, name, keys):
        client = self.getClient()
        return client.hdel(name, keys)

    def hgetall(self, name):
        client = self.getClient()
        return client.hgetall(name)

    def hset(self, name, key, value):
        client = self.getClient()
        return client.hset(name, key, value)

    def get(self, key):
        cache = self.getCache()
        if key in cache:
            # print '.............get cache:', key
            return cache[key]
        if hasattr(config, 'LOCAL_CACHE') and config.LOCAL_CACHE:
            for prefix in OutOfRedis:
                if key.startswith(prefix):
                    return None

        start = time.time()
        client = self.getClient()
        data = client.get(key)
        logging.info('redis get.......................key: %s, time: %f' % (key, time.time() - start))
        self.setCache(key, data)
        return data

    def set(self, key, value, time_limit=datetime.timedelta(days=365)):
        self.setCache(key, value)
        if hasattr(config, 'LOCAL_CACHE') and config.LOCAL_CACHE:
            for prefix in OutOfRedis:
                if key.startswith(prefix):
                    return None

        start = time.time()
        client = self.getClient()
        if time_limit is not None:
            result = client.set(key, value, time_limit)
            logging.info('redis set.......................key: %s, result: %s, time: %f' % (key, str(result), time.time() - start))
        else:
            result = client.set(key, value, datetime.timedelta(days=365))
            logging.info('redis set.......................key: %s, result: %s, time: %f' % (key, str(result), time.time() - start))
        return result

    def setForever(self, key, value):
        client = self.getClient()
        result = client.set(key, value)
        return result

    # 获得指定的key
    def keys(self, pattern):
        client = self.getClient()
        return client.keys(pattern)

    # 删除指定的key
    def delete(self, pattern):
        client = self.getClient()
        keys = client.keys(pattern)
        count = 0
        for key in keys:
            count += client.delete(key)
        return count

    # 真实的删除
    def delete_for_real(self, pattern):
        key_list = self.keys(pattern)
        if len(key_list) > 0:
            client = self.getClient()
            result = client.delete(*key_list)
            logging.info('redis delete.......................%s' % str(result))
            return result
        return 0

    # 获得客户端
    def getClient(self, db=None):
        if db is None:
            db = self.__db
        if not hasattr(config, 'REDIS_TYPE') or config.REDIS_TYPE == 'single':
            return redis.StrictRedis(connection_pool=self.__pool_dict[db])
        elif config.REDIS_TYPE == 'sentinel':
            master = self.__sentinel.master_for(config.REDIS_SENTINEL_MASTER_NAME, db=db, password=config.REDIS_PASSWORD)
            # master1 = self.__sentinel.discover_master('mymaster')
            # slave = mySentinel.slave_for("myslave", db=db, password=config.REDIS_PASSWORD)
            return master
        else:
            return None

    # 设置缓存
    def setCache(self, key, value):
        try:
            if hasattr(config, 'LOCAL_CACHE') and config.LOCAL_CACHE:
                global RedisCache
                cache = RedisCache
                if key is not None:
                    has_prefix = False
                    for prefix in NoCachePrefix:
                        if key.startswith(prefix):
                            has_prefix = True
                            break
                    if not has_prefix:
                        # print '.............set cache:', key
                        cache[key] = value
        except:
            pass

    def getCache(self):
        try:
            global RedisCache
            return RedisCache
        except:
            pass
        return {}


if __name__ == "__main__":
    util = RedisUtils(db=0)
    # util.setObject('test-key', 'hello world!', time_limit=datetime.timedelta(seconds=15), compress=False)
    # print util.getObject('test-key', compress=False)
    print(len(util.getFloatArray('ann-vec-mytestid')) / 256)