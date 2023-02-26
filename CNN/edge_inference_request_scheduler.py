import argparse
import requests
import roundrobin
from numpy import random
import time
from threading import Thread

parser = argparse.ArgumentParser()
parser.add_argument('--port', required=True, type=int)

args = parser.parse_args()

port = args.port

edges_info = {'nvidia-xavier2': {'url': f'http://192.168.0.30:{port}/',
                                 'model': ['mobilenet', 'mobilenet_v2', 'inception_v3']
                                 },
              'nvidia-tx2': {'url': f'http://192.168.0.8:{port}/',
                             'model': ['mobilenet', 'mobilenet_v2', 'inception_v3']
                             },
              'nvidia-nano1': {'url': f'http://192.168.0.29:{port}/',
                               'model': ['mobilenet']
                               }
              }


def model_request(edge, model, order):
    req_processing_start_time = time.time()
    edge_info = edges_info.get(edge)
    url = edge_info.get('url') + model
    res = requests.get(url)
    processing_time = time.time() - req_processing_start_time
    print(f'[{order}] total request time: {processing_time}\n{res.text}')

    return


model_edge_info = {}

for edge_info_key in edges_info.keys():
    edge_info = edges_info.get(edge_info_key)
    for model in edge_info.get('model'):
        if model not in model_edge_info.keys():
            model_edge_info[model] = []

        model_edge_info[model].append((edge_info_key, 1))

for i in model_edge_info.keys():
    model_info = model_edge_info.get(i)
    model_edge_info[i] = roundrobin.smooth(model_info)


requests_list = ['mobilenet', 'mobilenet_v2', 'inception_v3', 'mobilenet_v2', 'inception_v3', 'mobilenet']

threads = []
order = 0
for req in requests_list:
    edge_to_inference = model_edge_info.get(req)()
    order += 1
    th = Thread(target=model_request, args=(edge_to_inference, req, order))
    th.start()
    threads.append(th)

for th in threads:
    th.join()

# events_avg = 10
# total_event_num = 10
#
# poisson_distribution = random.poisson(events_avg, total_event_num)
#
# for event_num in poisson_distribution:
#     print('request count: ', event_num)
#
#     for idx in range(event_num):
#         current_inference_model = get_weighted_smooth()
#
#         request_start_time = time.time()
#         res = model_request(current_inference_model)
#         print('total request time: ', time.time() - request_start_time)
#
#         print(res)
