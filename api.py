import torch
import argparse
import model
from flask import Flask, request, make_response, jsonify
from model import textcnn, birnn
from dataset import TEXT, LABEL
from utils import transform_data

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

parser = argparse.ArgumentParser()
parser.add_argument('--model-name', default='birnn',choices=['textcnn', 'birnn'], help='choose one model name for trainng')
parser.add_argument('-lmd', '--load-model-dir', default= None, help='path for loadding model, default:None' )
args = parser.parse_args()

# 获取模型名称
net = getattr(model, args.model_name)()
net.load_state_dict(torch.load(args.load_model_dir))

# net = birnn()
# net.load_state_dict(torch.load('models_storage/model_birnn.pt'))

@app.route('/sentiment')
def sentiemnt():
    sentence = request.args.get('sentence')
    record = {'data':sentence}
    data, _ = transform_data(record, TEXT, LABEL)
    prediction = net(data).argmax(dim=1).item()
    if prediction==0:
        result = '积极'
    else:
        result = '消极'
    return jsonify({'data':result, 'status_code':200})


if __name__ == '__main__':
    app.run(debug=False)

