import os
import json
import pandas as pd

from variables import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from model import DDImodel
import logging
logging.getLogger('tensorflow').disabled = True

from util import get_data

from flask import Flask
from flask import jsonify
from flask import request

'''
        python -W ignore app.py
'''

app = Flask(__name__)

ddi = DDImodel()
ddi.dnn()

@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    A_drug = message['A_drug']
    B_drug = message['B_drug']
    ddi_value = ddi.predictions(int(A_drug), int(B_drug))
    response = {
            'ddi_value': int(ddi_value)
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True, host=host, port= port, threaded=False)