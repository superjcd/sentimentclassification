import torch
from flask import Flask, request, make_response, jsonify
from model import textcnn, birnn
from dataset import TEXT, LABEL
from utils import transform_data

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

net = birnn()
net.load_state_dict(torch.load('models_storage/model_birnn.pt'))

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

