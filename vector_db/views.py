import os
import time
import math
import json
import glob

from .utils import *
from .vector_extractor_v2 import Yolov3
from .elasticsearch import Elk
from .img_to_vec import imgtovec

from django.shortcuts import render, HttpResponse
from django.views.decorators.csrf import csrf_exempt


YOLO = Yolov3
ELK = Elk
ELK.save_index = 'data_'
ELK.search_index = 'data_'
ITV = imgtovec


def index(request):
    return render(request, 'vector_db/index.html')


@csrf_exempt
def search_vector_service(request):
    search_time = time.time()

    rb_list = []
    es_list = []
    result_dict = {}

    img_b64 = request.POST['img_b64']
    gender = request.POST['person']

    #face_box = request.POST['f_box']
    vec = YOLO.vector_extraction_service(YOLO, base_img=img_b64)

    if len(vec) == 0:
        print('no box detect')
        return HttpResponse("-1")

    for idx in list(vec.keys()):
        for count in range(len(vec[idx])):
            print(idx, '>>>', count)
            vector = vec[idx][count]['feature_vector']
            rb = vec[idx][count]['raw_box']
            vector_bs = encode_array(vector)

            res = ELK.search_vec(ELK, idx, vector_bs)
            es_list.append(res)
            rb_list.append(rb)
    print('Interface time for searching vec', time.time() - search_time)

    result_dict['es'] = es_list
    result_dict['raw_box'] = rb_list

#    return HttpResponse(json.dumps(result_dict))
    return HttpResponse(time.time()-search_time)




@csrf_exempt
def save_vector_ela(request):
    sql = 'SELECT * FROM product_list WHERE status=1 ORDER BY cre_tt DESC LIMIT 10'
    product_list = ITV.connect_db(ITV,sql)

    print('Total count of product list',len(product_list))

    data_dict = {}
    img_path_list = []

    save_time = time.time()

    for idx, product in enumerate(product_list):
        line = list(product)
        img_path = ITV.base_img_path + line[6] + os.sep + line[7]
        if not os.path.isfile(img_path): continue
        img_path_list.append(img_path)
        data_dict[img_path] = [line[2], line[4], line[5], img_path, line[8], line[11]]

        if len(img_path_list) % 10 == 0:
            print(idx,'##########################ing')
            batch_size = 500
            n = 0
            for size in range(math.ceil(len(img_path_list) / batch_size)):
                total_vec = {}

                bulk_path = img_path_list[batch_size * n:batch_size * (n + 1)]
                n += 1
                vec = YOLO.vector_extraction_batch(YOLO, data=bulk_path, batch_size=100)
                for idx in vec.keys():
                    if idx in total_vec.keys():
                        total_vec[idx] = total_vec[idx] + vec[idx]
                    else:
                        total_vec[idx] = vec[idx]

            ELK.bulk_api(ELK, total_vec, data_dict)

            data_dict = {}
            img_path_list = []

    print('Interface time for saving all data('+str(len(product_list))+')', time.time() - save_time)

    return HttpResponse('send to elk')


@csrf_exempt
def put_database(request):
    bulk_path = glob.glob('database/*')
    batch_size = 50
    vec = ITV.vector_extract(batch=True, data=bulk_path, batch_size=batch_size)
    ELK.bulkAPI_test(vec)
    return HttpResponse('submit')
