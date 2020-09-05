import os
import json
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger('tensorflow').disabled = True

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

from tensorflow import keras
if tf.config.list_physical_devices('GPU'):
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from keras.models import model_from_json, load_model
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Dropout, concatenate
from keras.models import Sequential, Model
from variables import *
from util import get_data
from collections import Counter

class DDImodel(object):
    def __init__(self):
        X, Y = get_data()
        Xssp, Xtsp, Xgsp = X
        self.Xssp = Xssp
        self.Xtsp = Xtsp
        self.Xgsp = Xgsp
        self.Y = Y

    @staticmethod
    def autoencoder(X, input_dim, autoencoder_weights):

        if os.path.exists(autoencoder_weights):
            print(" {} model Loading !!!".format(autoencoder_weights.split('/')[1][:-3]))
            autoencoder_model = load_model(autoencoder_weights)
            autoencoder_model.compile(
                                    optimizer='adam',
                                    loss='binary_crossentropy'
                                    )
        else:
            print(" {} model Training !!!".format(autoencoder_weights.split('/')[1][:-3]))
            inputs = Input(shape=(input_dim,))
            x = Dense(dim1, activation='relu')(inputs)
            x = Dense(dim2, activation='relu')(x)
            x = Dense(dim3, activation='relu')(x)
            x = Dense(dim2, activation='relu')(x)
            x = Dense(dim1, activation='relu')(x)
            output = Dense(input_dim, activation='relu')(x)

            autoencoder_model = Model(inputs, output)

            autoencoder_model.compile(
                                    optimizer='adam',
                                    loss='binary_crossentropy'
                                    )

            autoencoder_model.fit(  X,
                                    X,
                                    epochs=n_epochs,
                                    batch_size=batch_size,
                                    validation_split=val_split,
                                    verbose = 0
                                    )
            autoencoder_model.save(autoencoder_weights)

        return autoencoder_model

    @staticmethod
    def encoder(input_dim, autoencoder_model):
        inputs = Input(shape=(input_dim,))
        layer1 = Dense(dim1, activation='relu', trainable=False, weights=autoencoder_model.layers[1].get_weights())(inputs)
        layer2 = Dense(dim2, activation='relu', trainable=False, weights=autoencoder_model.layers[2].get_weights())(layer1)
        layer3 = Dense(dim3, activation='relu', trainable=False, weights=autoencoder_model.layers[3].get_weights())(layer2)

        return inputs, layer3

    def train_autoencoder(self):
        ssp_autoencoder = DDImodel.autoencoder(self.Xssp, n_ssp, ssp_weights)
        tsp_autoencoder = DDImodel.autoencoder(self.Xtsp, n_tsp, tsp_weights)
        gsp_autoencoder = DDImodel.autoencoder(self.Xgsp, n_gsp, gsp_weights)

        return ssp_autoencoder, tsp_autoencoder, gsp_autoencoder

    @staticmethod
    def get_encoder_output(ssp_autoencoder, tsp_autoencoder, gsp_autoencoder):
        ssp_inputs, ssp_out = DDImodel.encoder(n_ssp, ssp_autoencoder)
        tsp_inputs, tsp_out = DDImodel.encoder(n_tsp, tsp_autoencoder)
        gsp_inputs, gsp_out = DDImodel.encoder(n_gsp, gsp_autoencoder)

        return ssp_inputs, tsp_inputs, gsp_inputs, ssp_out, tsp_out, gsp_out

    def build_dnn_input(self):
        ssp_autoencoder, tsp_autoencoder, gsp_autoencoder = self.train_autoencoder()
        ssp_inputs, tsp_inputs, gsp_inputs, ssp_out, tsp_out, gsp_out = DDImodel.get_encoder_output(ssp_autoencoder, tsp_autoencoder, gsp_autoencoder)

        merged = concatenate([ssp_out, tsp_out])
        dnn_input = concatenate([merged,  gsp_out])
        # dnn_input = concatenate([ssp_out, tsp_out,  tsp_out])
        return ssp_inputs, tsp_inputs, gsp_inputs, dnn_input

    def dnn(self):
        if os.path.exists(dnn_weights):
            print("\n Final Model Loading !!!")
            dnn_model = load_model(dnn_weights)
            dnn_model.compile(
                            optimizer='adam',
                            loss='binary_crossentropy'
                            )
        else:
            ssp_inputs, tsp_inputs, gsp_inputs, dnn_input = self.build_dnn_input()

            print("\n Final Model Training !!!")
            x = Dense(dense1, activation='relu')(dnn_input)
            x = Dense(dense2, activation='relu')(x)
            x = Dense(dense3, activation='relu')(x)
            output = Dense(dense4, activation='sigmoid')(x)

            dnn_model = Model(inputs = [ssp_inputs, tsp_inputs, gsp_inputs],
                              outputs= output,
                              name='FInal DNN')

            dnn_model.summary()

            dnn_model.compile(
                            optimizer='adam',
                            loss='binary_crossentropy'
                            )

            dnn_model.fit(  [self.Xssp, self.Xtsp, self.Xgsp],
                            self.Y,
                            epochs=dnn_epoches,
                            batch_size=batch_size,
                            validation_split=val_split
                                    )
            dnn_model.save(dnn_weights)
        self.dnn_model = dnn_model

    def predictions(self, drug_id):
        X = [[self.Xssp[drug_id]], [self.Xtsp[drug_id]], [self.Xgsp[drug_id]]]
        Y = self.Y[drug_id]
        P = (self.dnn_model.predict(X).squeeze() > threshold)
        return P