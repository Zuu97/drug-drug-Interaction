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
# from apscheduler.schedulers.background import BackgroundScheduler
# scheduler = BackgroundScheduler()

'''
        python -W ignore app.py
'''

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    drug_id = message['drug_id']
    ddi_value = 0
    response = {
            'ddi_value': ddi_value
    }
    return jsonify(response)

if __name__ == "__main__":
    ddi = DDImodel()
    ddi.dnn()
    app.run(debug=True, host=host, port= port, threaded=False)