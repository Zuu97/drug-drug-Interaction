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
        XsspA, XtspA, XgspA, XsspB, XtspB, XgspB = X
        self.XsspA = XsspA
        self.XtspA = XtspA
        self.XgspA = XgspA
        self.XsspB = XsspB
        self.XtspB = XtspB
        self.XgspB = XgspB
        self.Y = Y

    @staticmethod
    def autoencoder(XA, XB, input_dim, autoencoder_weights):

        if os.path.exists(autoencoder_weights):
            print(" {} model Loading !!!".format(autoencoder_weights.split('/')[1][:-3]))
            autoencoder_model = load_model(autoencoder_weights)
            autoencoder_model.compile(
                                    optimizer='adam',
                                    loss='binary_crossentropy'
                                    )
        else:
            print(" {} model Training !!!".format(autoencoder_weights.split('/')[1][:-3]))
            inputA = Input(shape=(input_dim,))
            inputB = Input(shape=(input_dim,))
            inputs = concatenate([inputA, inputB])
            x = Dense(dim1, activation='relu')(inputs)
            x = Dense(dim2, activation='relu')(x)
            x = Dense(dim3, activation='relu')(x)
            x = Dense(dim2, activation='relu')(x)
            x = Dense(dim1, activation='relu')(x)
            output = Dense(2*input_dim, activation='relu')(x)

            autoencoder_model = Model(inputs=[inputA, inputB],
                                      outputs=output)

            autoencoder_model.compile(
                                    optimizer='adam',
                                    loss='binary_crossentropy'
                                    )

            X = np.concatenate((XA, XB), axis=1)

            autoencoder_model.fit(  [XA, XB],
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
        inputA = Input(shape=(input_dim,))
        inputB = Input(shape=(input_dim,))
        inputs = concatenate([inputA, inputB])
        layer1 = Dense(dim1, activation='relu', trainable=False, weights=autoencoder_model.layers[3].get_weights())(inputs)
        layer2 = Dense(dim2, activation='relu', trainable=False, weights=autoencoder_model.layers[4].get_weights())(layer1)
        layer3 = Dense(dim3, activation='relu', trainable=False, weights=autoencoder_model.layers[5].get_weights())(layer2)

        return inputA, inputB, layer3

    def train_autoencoder(self):
        ssp_autoencoder = DDImodel.autoencoder(self.XsspA, self.XsspB, n_ssp, ssp_weights)
        tsp_autoencoder = DDImodel.autoencoder(self.XtspA, self.XtspB, n_tsp, tsp_weights)
        gsp_autoencoder = DDImodel.autoencoder(self.XgspA, self.XgspB, n_gsp, gsp_weights)

        return ssp_autoencoder, tsp_autoencoder, gsp_autoencoder

    @staticmethod
    def get_encoder_output(ssp_autoencoder, tsp_autoencoder, gsp_autoencoder):
        ssp_inputsA, ssp_inputsB, ssp_out = DDImodel.encoder(n_ssp, ssp_autoencoder)
        tsp_inputsA, tsp_inputsB, tsp_out = DDImodel.encoder(n_tsp, tsp_autoencoder)
        gsp_inputsA, gsp_inputsB, gsp_out = DDImodel.encoder(n_gsp, gsp_autoencoder)

        inputsA = ssp_inputsA, tsp_inputsA, gsp_inputsA
        inputsB = ssp_inputsB, tsp_inputsB, gsp_inputsB
        outputs = ssp_out, tsp_out, gsp_out
        return inputsA, inputsB, outputs

    def build_dnn_input(self):
        ssp_autoencoder, tsp_autoencoder, gsp_autoencoder = self.train_autoencoder()
        inputsA, inputsB, outputs = DDImodel.get_encoder_output(ssp_autoencoder, tsp_autoencoder, gsp_autoencoder)

        ssp_out, tsp_out, gsp_out = outputs

        merged = concatenate([ssp_out, tsp_out])
        dnn_input = concatenate([merged,  gsp_out])
        return inputsA, inputsB, dnn_input

    def dnn(self):
        if os.path.exists(dnn_weights):
            print("\n Final Model Loading !!!")
            dnn_model = load_model(dnn_weights)
            dnn_model.compile(
                            optimizer='adam',
                            loss='sparse_categorical_crossentropy'
                            )
        else:
            inputsA, inputsB, dnn_input = self.build_dnn_input()

            ssp_inputsA, tsp_inputsA, gsp_inputsA = inputsA
            ssp_inputsB, tsp_inputsB, gsp_inputsB = inputsB

            print("\n Final Model Training !!!")
            x = Dense(dense1, activation='relu')(dnn_input)
            x = Dense(dense2, activation='relu')(x)
            x = Dense(dense3, activation='relu')(x)
            output = Dense(dense4, activation='softmax')(x)

            dnn_model = Model(inputs = [ssp_inputsA, ssp_inputsB, tsp_inputsA, tsp_inputsB, gsp_inputsA, gsp_inputsB],
                              outputs= output,
                              name='Final DNN')

            dnn_model.summary()

            dnn_model.compile(
                            optimizer='adam',
                            loss='sparse_categorical_crossentropy'
                            )

            dnn_model.fit(  [self.XsspA, self.XsspB, self.XtspA, self.XtspB, self.XgspA, self.XgspB],
                            self.Y,
                            epochs=dnn_epoches,
                            batch_size=batch_size,
                            validation_split=val_split
                                    )
            dnn_model.save(dnn_weights)
        self.dnn_model = dnn_model

    def predictions(self, drug_pair_id):
        X = [[self.XsspA[drug_pair_id]],
            [self.XsspB[drug_pair_id]],
            [self.XtspA[drug_pair_id]],
            [self.XtspB[drug_pair_id]],
            [self.XgspA[drug_pair_id]],
            [self.XgspB[drug_pair_id]]]

        # X = [self.XsspA[drug_pair_id:drug_pair_id+100],
        #     self.XsspB[drug_pair_id:drug_pair_id+100],
        #     self.XtspA[drug_pair_id:drug_pair_id+100],
        #     self.XtspB[drug_pair_id:drug_pair_id+100],
        #     self.XgspA[drug_pair_id:drug_pair_id+100],
        #     self.XgspB[drug_pair_id:drug_pair_id+100]]

        Y = self.Y[drug_pair_id]
        P = self.dnn_model.predict(X).squeeze().argmax(axis=-1)
        return P