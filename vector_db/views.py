from django.shortcuts import render,HttpResponse
from django.http import JsonResponse

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from elasticsearch import Elasticsearch

from django.views.decorators.csrf import csrf_exempt

import time
from datetime import datetime

import os
import csv
import base64
import numpy as np
import time
import pymysql
import json
import urllib.request

from .vector_extractor_v2 import Yolov3

# Create your views here.

def index(request):
    return HttpResponse('django vector api server test')

def search_process(self):
    st_time = time.time()
    ITV = ImageToVector()

    path = 'https://09imgmini.eastday.com/mobile/20200211/20200211011603_cb30a42d261f935fa2ca5ba3c2aa921d_2.jpeg'
    save_path = '/home/searchimg/'+path.split('/')[-1]
    urllib.request.urlretrieve(path,save_path)

    vec = ImageToVector.vector_extract(ITV, batch=False, data=save_path)

    if len(vec) == 0:
        print('no box')

    index = list(vec.keys())[0]
    vector = vec[index][0]['feature_vector']
    vector_bs = ImageToVector.encode_array(vector)

    res = ImageToVector.searchVec(ITV, index, vector_bs)
    end_time = time.time()-st_time
    print(end_time)
    return HttpResponse(res)

def all_process(self, bulk=True):
    sql = 'SELECT * FROM product_list LIMIT 2000000'
    #sql = 'SELECT * FROM product_list LIMIT 200'
    ITV = ImageToVector()
    product_list = ITV.connect_db_get_Date(sql)

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

    else: # not bulk
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

    es = Elasticsearch('49.247.197.215:9200')

    @classmethod
    def decode_float_list(self, base64_string):
        bytes = base64.b64decode(base64_string)
        return np.frombuffer(bytes, dtype=self.dfloat32).tolist()

    @classmethod
    def encode_array(self, arr):
        base64_str = base64.b64encode(np.array(arr).astype(self.dfloat32)).decode("utf-8")
        return base64_str

    @classmethod
    def utc_time(self):  # @timestamp timezone을 utc로 설정하여 kibana로 index 생성시 참조
        # return datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        return datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'


    def connect_db_get_Date(self, sql):
        print('Query to mysql......!!')
        conn = pymysql.connect(host='piclick-hk-ai.mariadb.rds.aliyuncs.com', user='piclick', password='psr0035!',
                               db='piclick')
        #sql = 'SELECT * FROM product_list LIMIT 100'
        st_time = time.time()
        curs = conn.cursor()
        curs.execute(sql)
        data = curs.fetchall()
        end_time = time.time()-st_time
        print('sql query time is....', end_time)
        # ['id','au_id','p_key','p_category','img_url','click_url','save_path','save_name','status','imp_cnt','click_cnt','cre_tt','appr_tt','rej_tt']

        curs.close()
        conn.close()
        return data

    def vector_extract(self, batch=False, data='/home/piclick/vector_api/test_img/',batch_size=100):
        yolov3 = Yolov3()

        start_time = time.time()
        if batch:
            res = yolov3.vector_extraction_batch(bulk_path=data,batch_size=batch_size)
        else:
            res = yolov3.vector_extraction_one_img(data)
        end_time = time.time() - start_time

        return res

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

        index = "vector_3nd_" + index.lower()

        with open('/home/vector_api/vector_db/mapping.json') as f:
            mapping = json.load(f)

            #test
        try:
            self.es.indices.create(index=index, body=mapping)
        except:
            pass
        finally:
            self.es.index(index=index, doc_type="_doc", body=doc)


    def searchVec(self, search_index, search_vec):

        search_index = 'vector_'+ search_index.lower()

        res = self.es.search(
                index = search_index,
                body  = {
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
                    "size": 10
                }
        )

        return json.dumps(res, ensure_ascii=True, indent='\t')




