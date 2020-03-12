import json
import time
from elasticsearch import Elasticsearch, helpers
from .utils import *


class Elk:
    es = Elasticsearch('49.247.197.215:9200')

    search_index = 'vector_'
    save_index = 'vector_'

    def create_index(self, index):
        with open('/vector_api/vector_db/mapping.json') as f:
            mapping = json.load(f)

        if type(index) == list:
            for idx in index:
                if not self.es.indices.exists(index=idx):
                    self.es.indices.create(index=idx, body=mapping)
        else:
            if not self.es.indices.exists(index=index):
                self.es.indices.create(index=index, body=mapping)

    def bulk_api(self, total_vec, data_dict):
        docs = []
        es_time = time.time()
        for v_idx in total_vec.keys():
            new_index = self.save_index + v_idx.lower()
            self.create_index(self, new_index)
            print(v_idx, '>>', len(total_vec[v_idx]))
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
                        "vector_bs": encode_array(total_vec[v_idx][count]['feature_vector']),
                        "@timestamp": utc_time()
                    }
                })
        helpers.bulk(self.es, docs)
        print('Interface time for sending to elastic',time.time() - es_time)

    def bulk_api_test(self, total_vec):
        docs = []
        es_time = time.time()
        for v_idx in total_vec.keys():
            new_index = 'vector_' + v_idx.lower()
            self.createIndex(new_index)
            print(v_idx, '>>', len(total_vec[v_idx]))
            for count in range(len(total_vec[v_idx])):
                docs.append({
                    '_index': new_index,
                    '_source': {
                        "p_key": "test",
                        "img_url": "test",
                        "click_url": "test",
                        "image_path": "test",
                        "status": 3,
                        "cre_tt": utc_time(),
                        "raw_box": np.array(total_vec[v_idx][count]['raw_box']).tolist(),
                        "vector_bs": encode_array(total_vec[v_idx][count]['feature_vector']),
                        "@timestamp": utc_time()
                    }
                })

        helpers.bulk(self.es, docs)

        print('Interface time for sending to elastic',time.time() - es_time)

    def search_vec(self, search_index, search_vec):
        st_time = time.time()
        search_index = self.search_index + search_index.lower()
        res = self.es.search(
            index=search_index,
            body={
                "_source": {
                    "includes": ["_index", "_score", "", "p_key", "img_url", "click_url", "image_path", "raw_box"]
                },
                "query": {"function_score": {
                    "boost_mode": "replace",
                    "script_score": {"script": {
                        "source": "binary_vector_score", "lang": "knn",
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
        print('Interface time for searching vector', time.time() - st_time)

        return json.dumps(res, ensure_ascii=True, indent='\t')