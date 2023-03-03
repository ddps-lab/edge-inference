import argparse
import requests
import roundrobin
from numpy import random
import time
from threading import Thread

parser = argparse.ArgumentParser()
parser.add_argument('--edge', default=None, type=str)
parser.add_argument('--port', default=5001, type=int)

######### 임시 코드 ###################
parser.add_argument('--reqs', default='mobilenet,10', type=str)
parser.add_argument('--random', action='store_true')

temp_args = parser.parse_args()

inference_requests = temp_args.reqs.split(',')
inference_random_flag = temp_args.random
####################################

args = parser.parse_args()

edges_to_inference = args.edge
port = args.port


# 이 부분만 설정하면 모델추가나 장비추가가 수월함. 각 장비의 ip와 로드된 모델들을 설정해주어야함.
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


# --edge 옵션이 없을 시 등록되어 있는 모든 장비들에 추론 요청, 요청장비들은 edges_info에 등록되어 있어야함. 입력 형식은 'a, b, ...'
edges_register = list(edges_info.keys())

if edges_to_inference is None:
    edges_to_inference = edges_register
else:
    edges_to_inference = edges_to_inference.split(',')

for edge in edges_to_inference:
    if edge not in edges_register:
        print(f'--edge arg must be in {edges_register}')
        exit(1)

print(f'Edges to inference: {edges_to_inference}')


# 추론 요청 할 장비들에서 요청 가능한 모델들
models_to_inference = []

for edge_name in edges_to_inference:
    edge_info = edges_info.get(edge_name)
    models = edge_info.get('model')
    models_to_inference.extend(models)

models_to_inference = set(models_to_inference)

print(f'Models to inference: {models_to_inference}')


# 추론을 요청하는 함수, 인자로는 추론을 요청할 엣지 장비, 모델, 요청임. 엣지장비와 모델은 위의 edges_info에 등록되어 있어야함
def model_request(edge, model, order):
    if edge not in edges_to_inference:
        print(f'[{order}] edge must be in {edges_to_inference}/ input value: {edge}')
        return

    if model not in models_to_inference:
        print(f'[{order}] model must be in {models_to_inference}/ input value: {model}')
        return

    req_processing_start_time = time.time()
    edge_info = edges_info.get(edge)
    url = edge_info.get('url') + model
    res = requests.get(url)
    processing_time = time.time() - req_processing_start_time
    print(f'[{order}] total request time: {processing_time}\n{res.text}')
    return


# 현재 스케줄링 방식: 딕셔너리에 모델별로 엣지장비이름 등록, 들어오는 요청에 따라 각 장비들에 라운드로빈으로 스케줄링
# 문제점 각 모델이 하나씩 들어오면 장비들 중 하나에만 요청이 들어감 -> 고민
model_edge_info = {}

for edge in edges_to_inference:
    edge_info = edges_info.get(edge)
    for model in edge_info.get('model'):
        if model not in model_edge_info.keys():
            model_edge_info[model] = []

        model_edge_info[model].append((edge, 1))

print(f'model-edge dataset: {model_edge_info}')


for model in model_edge_info.keys():
    dataset = model_edge_info.get(model)
    model_edge_info[model] = roundrobin.smooth(dataset)


# 들어오는 요청들 임시 코드임!!!
requests_list = []
for idx in range(0, len(inference_requests), 2):
    model = inference_requests[idx]
    inference_num = int(inference_requests[idx+1])

    for _ in range(inference_num):
        requests_list.append(model)

if inference_random_flag:
    random.shuffle(requests_list)


threads = []
order = 0
for req in requests_list:
    if req in model_edge_info.keys():
        edge_to_inference = model_edge_info.get(req)()
    else:
        edge_to_inference = ''

    order += 1
    th = Thread(target=model_request, args=(edge_to_inference, req, f'{order}:{edge_to_inference}/{req}'))
    th.start()
    threads.append(th)

for th in threads:
    th.join()
