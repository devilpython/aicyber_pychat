# -*- coding: UTF-8 -*-
import hashlib

def get_md5_value(src):
    src = src.encode('utf8')
    myMd5 = hashlib.md5()
    myMd5.update(src)
    myMd5_Digest = myMd5.hexdigest()
    return myMd5_Digest

def get_sha1_value(src):
    src = src.encode('utf8')
    mySha1 = hashlib.sha1()
    mySha1.update(src)
    mySha1_Digest = mySha1.hexdigest()
    return mySha1_Digest

if __name__ == '__main__':
    print(get_md5_value('你好'))