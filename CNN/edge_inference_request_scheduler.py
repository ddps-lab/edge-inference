import argparse
import requests

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='mobilenet,mobilenet_v2,inception_v3', type=str)
parser.add_argument('--hostname', required=True, type=str)
parser.add_argument('--port', required=True, type=int)

args = parser.parse_args()

models_to_inference = args.model.split(',')
hostname = args.hostname
port = args.port

inference_request_url = f'http://{hostname}:{port}/'


def model_reqest(model):
    url = inference_request_url + model
    res = requests.post(url)

    return res.text


for model in models_to_inference:
    res = model_reqest(model)
    print(res)
