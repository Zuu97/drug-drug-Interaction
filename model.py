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
import tensorflow as tf
if tf.config.list_physical_devices('GPU'):
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from keras.models import model_from_json, load_model
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Dropout, Concatenate
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

    def autoencoder(self, X, input_dim, autoencoder_weights):

        if os.path.exists(autoencoder_weights):
            autoencoder_model = load_model(autoencoder_weights)
            autoencoder_model.compile(
                                    optimizer='adam',
                                    loss='binary_crossentropy'
                                    )
        else:
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
                                    validation_split=val_split
                                    )
            autoencoder_model.save(autoencoder_weights)

        return autoencoder_model

    def train_encoders(self):
        ssp_decoder = self.autoencoder(self.Xssp, n_ssp, ssp_weights)
        tsp_decoder = self.autoencoder(self.Xtsp, n_tsp, tsp_weights)
        gsp_decoder = self.autoencoder(self.Xgsp, n_gsp, gsp_weights)
        return ssp_decoder, tsp_decoder, gsp_decoder

    def build_dnn_input(self):
        ssp_decoder, tsp_decoder, gsp_decoder = self.train_encoders()
        ssp_encoder_layer = ssp_decoder.layers[-4]
        tsp_encoder_layer = tsp_decoder.layers[-4]
        gsp_encoder_layer = gsp_decoder.layers[-4]

        merged = Concatenate([ssp_encoder_layer, tsp_encoder_layer])
        dnn_input = Concatenate([merged,  gsp_encoder_layer])
        return dnn_input

    def dnn(self, Xencode):
        input_dim = self.build_dnn_input()

        if os.path.exists(dnn_weights):
            dnn_model = load_model(dnn_weights)
            dnn_model.compile(
                            optimizer='adam',
                            loss='binary_crossentropy'
                            )
        else:
            inputs = Input(shape=(input_dim,))
            x = Dense(dense1, activation='relu')(inputs)
            x = Dense(dense2, activation='relu')(x)
            x = Dense(dense3, activation='relu')(x)
            output = Dense(dense4, activation='sigmoid')(x)

            dnn_model = Model(inputs, output)

            dnn_model.compile(
                            optimizer='adam',
                            loss='binary_crossentropy'
                            )

            dnn_model.fit(  X,
                            X,
                            epochs=n_epochs,
                            batch_size=batch_size,
                            validation_split=val_split
                                    )
            dnn_model.save(dnn_weights)

        return dnn_model