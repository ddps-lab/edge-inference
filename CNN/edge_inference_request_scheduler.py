import argparse
import requests
import roundrobin
from numpy import random
import time
from threading import Thread

parser = argparse.ArgumentParser()
parser.add_argument('--edge', default=None, type=str)

######### 임시 코드 ###################
parser.add_argument('--reqs', default='mobilenet,10', type=str)
parser.add_argument('--random', action='store_true')

temp_args = parser.parse_args()

inference_requests = temp_args.reqs.split(',')
inference_random_flag = temp_args.random
####################################

args = parser.parse_args()

edges_to_inference = args.edge


# 이 부분만 설정하면 모델추가나 장비추가가 수월함. 각 장비의 ip와 로드된 모델들을 설정해주어야함.
edges_info = {'nvidia-xavier2': {'ip_addr': '192.168.0.32',
                                 'ports': [5001, 5002],
                                 'models': ['mobilenet', 'mobilenet_v2', 'inception_v3', 'yolo_v5']
                                 },
              'nvidia-tx2': {'ip_addr': '192.168.0.22',
                             'ports': [5001],
                             'models': ['mobilenet', 'mobilenet_v2', 'inception_v3', 'yolo_v5']
                             },
              'nvidia-nano1': {'ip_addr': '192.168.0.41',
                               'ports': [5001],
                               'models': ['mobilenet']
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
    models = edge_info.get('models')
    models_to_inference.extend(models)

models_to_inference = set(models_to_inference)

print(f'Models to inference: {models_to_inference}')


# 현재 스케줄링 방식: 딕셔너리에 모델별로 엣지장비이름 등록, 들어오는 요청에 따라 각 장비들에 라운드로빈으로 스케줄링
# 문제점 각 모델이 하나씩 들어오면 장비들 중 하나에만 요청이 들어감 -> 고민
model_edge_info = {}

for edge in edges_to_inference:
    edge_info = edges_info.get(edge)
    for model in edge_info.get('models'):
        if model not in model_edge_info.keys():
            model_edge_info[model] = []

        model_edge_info[model].append((edge, 1))

print(f'model-edge dataset: {model_edge_info}')

for model in model_edge_info.keys():
    dataset = model_edge_info.get(model)
    model_edge_info[model] = roundrobin.smooth(dataset)


# 각 엣지 장비별 포트들을 라운드 로빈으로 얻어올 수 있도록 딕셔너리 구성
edge_port_info = {}

for edge in edges_to_inference:
    if edge not in edge_port_info.keys():
        edge_port_info[edge] = []

    edge_info = edges_info.get(edge)
    for port in edge_info.get('ports'):
        edge_port_info[edge].append((port, 1))

print(f'edge-port dataset: {edge_port_info}')

for edge in edge_port_info.keys():
    dataset = edge_port_info.get(edge)
    edge_port_info[edge] = roundrobin.smooth(dataset)


# 모델로 엣지장비 이름 얻는 함수, 모델마다 엣지장비들의 정보가 담겨 있고 얻어올 때는 라운드로빈으로 순서대로 가져옴
def get_edge_by_model_rr(model):
    if model in model_edge_info.keys():
        return model_edge_info.get(model)()
    else:
        return None


# 엣지장비 이름으로 해당 포트번호를 얻는 함수, 엣지장비마다 포트 번호 정보가 담겨 있고 라운드로빈으로 순서대로 가져옴
def get_port_by_edge_rr(edge):
    if edge in edge_port_info.keys():
        return edge_port_info.get(edge)()
    else:
        return None


# 추론을 요청하는 함수, 인자로는 추론을 요청할 엣지 장비, 모델, 요청임. 엣지장비와 모델은 위의 edges_info에 등록되어 있어야함
def model_request(edge, model, order):
    if edge not in edges_to_inference:
        print(f'[{order}] edge must be in {edges_to_inference}/ input value: {edge}')
        return

    if model not in models_to_inference:
        print(f'[{order}] model must be in {models_to_inference}/ input value: {model}')
        return

    edge_info = edges_info.get(edge)

    edge_ip_addr = edge_info.get('ip_addr')
    port = get_port_by_edge_rr(edge)
    url = f'http://{edge_ip_addr}:{port}/{model}'

    req_processing_start_time = time.time()
    res = requests.get(url)
    processing_time = time.time() - req_processing_start_time

    inference_time = res.text.split(':')[1]
    inference_time = inference_time.split('\n')[0]
    inference_time_results[order-1] = float(inference_time)
    request_time_results[order-1] = float(processing_time)

    print(f'[{order}:{edge}({port})/{model}] total request time: {processing_time}\n{res.text}')
    return


### 들어오는 요청들 임시 코드임!!! ###
requests_list = []
for idx in range(0, len(inference_requests), 2):
    model = inference_requests[idx]
    inference_num = int(inference_requests[idx+1])

    for _ in range(inference_num):
        requests_list.append(model)

if inference_random_flag:
    random.shuffle(requests_list)
##############################


# 요청을 각 장비에 전달, 여러요청을 동시에 다룰 수 있도록 쓰레드 이용
threads = []
order = 0

inference_time_results = [0 for _ in range(len(requests_list))]
request_time_results = [0 for _ in range(len(requests_list))]
request_sleep_time = 1 / len(requests_list)  # 요청들을 1초에 나눠서 보내기 위한 슬립시간

for req in requests_list:
    edge_to_inference = get_edge_by_model_rr(req)
    if edge_to_inference is None:
        print(f'{req} can\'t be inference')
        continue

    order += 1
    th = Thread(target=model_request, args=(edge_to_inference, req, order))
    th.start()
    threads.append(th)
    time.sleep(request_sleep_time)

for th in threads:
    th.join()


# 추론요청 결과 출력 (최소, 중간, 최대, 평균)
inference_time_results.sort()
len_inference_time_results = len(inference_time_results)
request_time_results.sort()
len_request_time_results = len(request_time_results)

total_inference_time = sum(inference_time_results)
avg_inference_time = total_inference_time / len_inference_time_results
min_inference_time = inference_time_results[0]
mid_inference_time = inference_time_results[int(len_inference_time_results / 2)]
max_inference_time = inference_time_results[-1]

total_request_time = sum(request_time_results)
avg_request_time = total_request_time / len_request_time_results
min_request_time = request_time_results[0]
mid_request_time = request_time_results[int(len_request_time_results / 2)]
max_request_time = request_time_results[-1]

print(f'평균 추론 시간: {avg_inference_time}')
print(f'최소 추론 시간: {min_inference_time}')
print(f'중간 추론 시간: {mid_inference_time}')
print(f'최대 추론 시간: {max_inference_time}\n')

print(f'평균 응답 시간: {avg_request_time}')
print(f'최소 응답 시간: {min_request_time}')
print(f'중간 응답 시간: {mid_request_time}')
print(f'최대 응답 시간: {max_request_time}\n')
