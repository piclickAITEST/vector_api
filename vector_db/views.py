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

from .vector_extractor_v2 import Yolov3

# Create your views here.

def index(request):
    return HttpResponse('django vector api server test')

def all_process(self):
    sql = 'SELECT * FROM product_list LIMIT 100'
    ITV = ImageToVector()
    product_list = ITV.connect_db_get_Date(sql)

    for product in product_list:
        line = list(product)
        img_path = ITV.base_img_path + line[6] + os.sep + line[7]
        img_path = '/home/piclick/data/content/images/train/media_201910_34094.jpeg'
        new_line = [line[2], line[4], line[5], img_path, line[8], line[11]]
        # new_line =  p_key, img_url, click_url, img_path, status, cre_tt

        vec = ITV.vector_extract(batch=False, data=img_path)
        if len(vec) == 0:
            continue

        # send to elk
        index = list(vec.keys())[0]

        for key in vec[index][0].keys():
            new_line.append(list(vec[index][0][key]))

        #new_line.append(list(vec[index][0]['raw_box']))
        #new_line.append(list(vec[index][0]['feature_vector_max']))

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
        conn = pymysql.connect(host='piclick-hk-ai.mariadb.rds.aliyuncs.com', user='piclick', password='psr0035!',
                               db='piclick')
        # sql = 'SELECT * FROM product_list LIMIT 100'
        curs = conn.cursor()
        curs.execute(sql)
        data = curs.fetchall()
        # ['id','au_id','p_key','p_category','img_url','click_url','save_path','save_name','status','imp_cnt','click_cnt','cre_tt','appr_tt','rej_tt']

        return data

    def vector_extract(self, batch=False, data='/home/piclick/vector_api/test_img/'):
        yolov3 = Yolov3()

        start_time = time.time()
        if batch:
            res = yolov3.vector_extraction_batch(data)
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

        res = self.es.index(index=index.lower(), doc_type="_doc", body=doc)
        print('upload to elk')

