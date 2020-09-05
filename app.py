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
ddi_value = ddi.predictions(100)

@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    drug_id = message['drug_id']
    response = {
            'ddi_value': ddi_value
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True, host=host, port= port, threaded=False)