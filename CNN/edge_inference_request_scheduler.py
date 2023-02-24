import argparse
import requests
import roundrobin
from numpy import random
import time

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='mobilenet,mobilenet_v2,inception_v3', type=str)
parser.add_argument('--hostname', required=True, type=str)
parser.add_argument('--port', required=True, type=int)

args = parser.parse_args()

models_to_inference = args.model.split(',')
hostname = args.hostname
port = args.port

inference_request_url = f'http://{hostname}:{port}/'


def model_request(model):
    url = inference_request_url + model
    res = requests.get(url)

    return res.text


models = [('mobilenet', 1), ('mobilenet_v2', 3), ('inception_v3', 6)]
get_weighted_smooth = roundrobin.smooth(models)

events_avg = 10
total_event_num = 10

poisson_distribution = random.poisson(events_avg, total_event_num)

for event_num in poisson_distribution:
    print('request count: ', event_num)

    for idx in event_num:
        current_inference_model = get_weighted_smooth()

        request_start_time = time.time()
        res = model_request(current_inference_model)
        print('total request time: ', time.time() - request_start_time)

        print(res)





