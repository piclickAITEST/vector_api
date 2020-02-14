import os
import csv
import time
import json
import glob
import base64
import pymysql
import numpy as np
from PIL import Image
import urllib.request
from datetime import datetime

from .vector_extractor_v2 import Yolov3

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from elasticsearch import Elasticsearch, helpers

from django.shortcuts import render,HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

def index(request):
    #return HttpResponse('django vector api server test')
    return render(request, 'vector_db/index.html')

@csrf_exempt
def put_database(request):
    ITV = ImageToVector()
    ELK = Elk()

    bulk_path = glob.glob('database/*')
    batch_size = 50
    vec = ITV.vector_extract(batch=True, data=bulk_path, batch_size=batch_size)
    ELK.bulkAPI_test(vec)
    return HttpResponse('submit')

@csrf_exempt
def search_process(request):
    st_time = time.time()
    ITV = ImageToVector()
    ELK = Elk()

    path = request.POST['img_b64']
    vec = ITV.vector_extract(batch=False, data=path)

    if len(vec) == 0:
        return HttpResponse('no box')

    indexes = list(vec.keys())
    rb_list = []
    es_list = []
    result_dict = {}
    for idx in indexes:
        for count in range(len(vec[idx])):
            print(idx, '>>>',count)
            vector = vec[idx][count]['feature_vector']
            rb = vec[idx][count]['raw_box']
            vector_bs = ITV.encode_array(vector)

            res = ELK.searchVec(idx, vector_bs)
            es_list.append(res)
            rb_list.append(rb)
    print(time.time()-st_time)

    result_dict['es'] = es_list
    result_dict['raw_box'] = rb_list

    return HttpResponse(json.dumps(result_dict))

def all_process(self, bulk=True):
    sql = 'SELECT * FROM product_list LIMIT 100000'

    ITV = ImageToVector()
    product_list = ITV.connect_db_get_Date(sql)

    '''
        if bulk: #ELASICBULK
            total_time = time.time()

            elk = Elk()

            data_dict = {}
            img_path_list = []
            for product in product_list:
                line = list(product)
                img_path = ITV.base_img_path + line[6] + os.sep + line[7]
                img_path_list.append(img_path)
                data_dict[img_path] = [line[2], line[4], line[5], img_path, line[8], line[11]]

            batch_size = 100
            n = 0
            total_vec = {}

            for size in range(int(len(img_path_list)/batch_size)):
                bulk_path = img_path_list[batch_size*n:batch_size*(n+1)]
                n += 1
                vec = ITV.vector_extract(batch=True,data = bulk_path,batch_size=batch_size)

                for idx in vec.keys():
                    if idx in total_vec.keys():
                        total_vec[idx] = total_vec[idx] + vec[idx]
                    else:
                        total_vec[idx] = vec[idx]

                elk.bulkAPI(total_vec, data_dict)
                elk_time = time.time()
                print(time.time()-elk_time,'~~~~~~~~~~')
            print(time.time()-total_time,'################################totaltime')
            '''

    if bulk:  # ELASICBULK
        total_time = time.time()

        elk = Elk()

        data_dict = {}
        img_path_list = []
        for product in product_list:
            line = list(product)
            img_path = ITV.base_img_path + line[6] + os.sep + line[7]
            img_path_list.append(img_path)
            data_dict[img_path] = [line[2], line[4], line[5], img_path, line[8], line[11]]

        batch_size = 100
        
        bulk_path = img_path_list
        vec = ITV.vector_extract(batch=True, data=bulk_path, batch_size=batch_size)

        elk.bulkAPI(vec, data_dict)
        print(time.time() - total_time, '################totaltime################')

    else:
        for product in product_list:
            line = list(product)
            img_path = ITV.base_img_path + line[6] + os.sep + line[7]
            #img_path = '/home/piclick/data/content/images/train/media_201910_34094.jpeg'
            new_line = [line[2], line[4], line[5], img_path, line[8], line[11]]
            # new_line =  p_key, img_url, click_url, img_path, status, cre_tt

            vec = ITV.vector_extract(batch=False, data=img_path)
            if len(vec) == 0:
                continue

            # send to elk
            index = list(vec.keys())[0]

            for key in vec[index][0].keys():
                new_line.append(list(vec[index][0][key]))

            ITV.vector2elk(index, new_line)
            # p_key, img_url, click_url, img_path, status, cre_tt, raw_box, vector
        print('done#############')
    
    return HttpResponse('upload data to elk')

class ImageToVector():
    dataFormat = "%Y-%m-%d %H:%M:%S"
    dfloat32 = np.dtype('>f4')
    base_img_path = "/mnt/piclick/piclick.tmp/AIMG/"

    def decode_float_list(self, base64_string):
        bytes = base64.b64decode(base64_string)
        return np.frombuffer(bytes, dtype=self.dfloat32).tolist()

    def encode_array(self, arr):
        base64_str = base64.b64encode(np.array(arr).astype(self.dfloat32)).decode("utf-8")
        return base64_str

    def utc_time(self):  # @timestamp timezone을 utc로 설정하여 kibana로 index 생성시 참조
        # return datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        return datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

    def connect_db_get_Date(self, sql):
        conn = pymysql.connect(host='piclick-hk-ai.mariadb.rds.aliyuncs.com', user='piclick', password='psr0035!',
                               db='piclick')
        st_time = time.time()
        curs = conn.cursor()
        curs.execute(sql)
        data = curs.fetchall()
        # ['id','au_id','p_key','p_category','img_url','click_url','save_path','save_name','status','imp_cnt','click_cnt','cre_tt','appr_tt','rej_tt']
        print('sql query time is....', time.time()-st_time)

        curs.close()
        conn.close()
        return data

    def vector_extract(self, batch=False, data='/home/piclick/vector_api/test_img/',batch_size=100):
        yolov3 = Yolov3()

        if batch:
            res = yolov3.vector_extraction_batch(bulk_path=data,batch_size=batch_size)
        else:
            #Service
            res = yolov3.vector_extraction_service(data)

        return res


class Elk():
    es = Elasticsearch('49.247.197.215:9200')

    def createIndex(self, index):
        with open('/vector_api/vector_db/mapping.json') as f:
            mapping = json.load(f)
        if type(index) == list:
            for idx in index:
                if not self.es.indices.exists(index=idx):
                    self.es.indices.create(index=idx, body=mapping)
        else:
            if not self.es.indices.exists(index=index):
                self.es.indices.create(index=index, body=mapping)

    def vector2elk(self, index, data):

        # p_key, img_url, click_url, img_path, status, cre_tt, raw_box, vector
        doc = {"p_key": data[0],
               "img_url": data[1],
               "click_url": data[2],
               "image_path": data[3],
               "status": data[4],
               "cre_tt": data[5],
               "raw_box": np.array(data[6]).tolist(),
               "vector_bs": self.encode_array(data[7]),
               "@timestamp": ImageToVector.utc_time()}

        index = "vector_" + index.lower()

        with open('/home/vector_api/vector_db/mapping.json') as f:
            mapping = json.load(f)

        if not self.es.indices.exists(index=index):
            self.es.indices.create(index=index, body=mapping)

        self.es.index(index=index, doc_type="_doc", body=doc)

    def bulkAPI(self, total_vec, data_dict):

        docs = []
        ITV = ImageToVector()
        es_time = time.time()
        for v_idx in total_vec.keys():
            new_index = 'test_' + v_idx.lower()
            self.createIndex(new_index)
            print(v_idx,'>>',len(total_vec[v_idx]))
            for count in range(len(total_vec[v_idx])):
                docs.append({
                    '_index': new_index,
                    '_source': {
                        "p_key": data_dict[total_vec[v_idx][count]['img_path']][0],
                        "img_url": data_dict[total_vec[v_idx][count]['img_path']][1],
                        "click_url": data_dict[total_vec[v_idx][count]['img_path']][2],
                        "image_path": data_dict[total_vec[v_idx][count]['img_path']][3],
                        "status": data_dict[total_vec[v_idx][count]['img_path']][4],
                        "cre_tt": data_dict[total_vec[v_idx][count]['img_path']][5],
                        "raw_box": np.array(total_vec[v_idx][count]['raw_box']).tolist(),
                        "vector_bs": ITV.encode_array(total_vec[v_idx][count]['feature_vector']),
                        "@timestamp": ITV.utc_time()
                    }
                })

        helpers.bulk(self.es, docs)

        print(time.time()-es_time,'------------------------------elk')

    def bulkAPI_test(self, total_vec):

        docs = []
        ITV = ImageToVector()
        es_time = time.time()
        for v_idx in total_vec.keys():
            new_index = 'vector_' + v_idx.lower()
            self.createIndex(new_index)
            print(v_idx,'>>',len(total_vec[v_idx]))
            for count in range(len(total_vec[v_idx])):
                docs.append({
                    '_index': new_index,
                    '_source': {
                        "p_key": "test",
                        "img_url": "test",
                        "click_url": "test",
                        "image_path": "test",
                        "status": 3,
                        "cre_tt": ITV.utc_time(),
                        "raw_box": np.array(total_vec[v_idx][count]['raw_box']).tolist(),
                        "vector_bs": ITV.encode_array(total_vec[v_idx][count]['feature_vector']),
                        "@timestamp": ITV.utc_time()
                    }
                })

        helpers.bulk(self.es, docs)

        print(time.time()-es_time,'------------------------------elk')

    def searchVec(self, search_index, search_vec):

        search_index = 'vector_'+ search_index.lower()
        st_time = time.time()
        res = self.es.search(
                index = search_index,
                body  = {
                          "_source": {
                             "includes": ["_index", "_score", "", "p_key", "img_url", "click_url", "image_path", "raw_box"]
                          },
                          "query": {"function_score": {
                              "boost_mode": "replace",
                              "script_score": {"script": {
                                 "source": "binary_vector_score","lang": "knn",
                                  "params": {
                                    "cosine": True,
                                    "field": "vector_bs",
                                    "encoded_vector": search_vec
                                  }
                                }
                              }
                            }
                          },
                    "size": 5
                },
            request_timeout=300
        )
        print(search_index,'>>>',time.time()-st_time)

        return json.dumps(res, ensure_ascii=True, indent='\t')






'''
    if bulk:
        st_time = time.time()
        data_dict = {}
        img_path_list = []
        for product in product_list:
            line = list(product)
            img_path = ITV.base_img_path + line[6] + os.sep + line[7]
            img_path_list.append(img_path)
            data_dict[img_path] = [line[2], line[4], line[5], img_path, line[8], line[11]]

        batch_size = 100
        n = 0

        total_box = 0
        for size in range(int(len(img_path_list)/batch_size)):
            print('#########',size,'#########')
            bulk_path = img_path_list[batch_size*n:batch_size*(n+1)]
            n += 1
            vec = ITV.vector_extract(batch=True,data = bulk_path,batch_size=batch_size)

            for index in vec.keys():
                total_box += len(vec[index])
                print(index, len(vec[index]))
                for count in range(len(vec[index])):
                    new_line = data_dict[vec[index][count]['img_path']][:6]

                    for key in list(vec[index][count].keys())[:-1]:
                        new_line.append(vec[index][count][key])

                    ITV.vector2elk(index, new_line)

            print('send all', time.time()-st_time)

        print('total box count', total_box)
    '''
