import os
import json
import pandas as pd
import matplotlib.pyplot as plt
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
from keras.optimizers import Adam
from variables import *
from util import get_data, get_prediction_data
from collections import Counter

from sqlalchemy import create_engine
import sqlalchemy

class DDImodel(object):
    def __init__(self):
        if not os.path.exists(dnn_weights):
            print(" Loading Data !!!")
            Xssp, Xtsp, Xgsp, Y = get_data()
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
            inputs = Input(shape=(2*input_dim,))
            x = Dense(dim1, activation='relu')(inputs)
            x = Dense(dim1, activation='relu')(x)
            x = Dense(dim2, activation='relu')(x)
            x = Dense(dim2, activation='relu')(x)
            x = Dense(dim3, activation='relu')(x)
            x = Dense(dim2, activation='relu')(x)
            x = Dense(dim2, activation='relu')(x)
            x = Dense(dim1, activation='relu')(x)
            x = Dense(dim1, activation='relu')(x)
            output = Dense(2*input_dim, activation='relu')(x)

            autoencoder_model = Model(inputs=inputs,
                                      outputs=output)

            autoencoder_model.compile(
                                    optimizer='adam',
                                    loss='binary_crossentropy'
                                    )


            history_auctoencoder = autoencoder_model.fit(   X,
                                                            X,
                                                            epochs=n_epochs,
                                                            batch_size=batch_size,
                                                            validation_split=val_split,
                                                            verbose = 0
                                                            )
            autoencoder_model.save(autoencoder_weights)
            DDImodel.plot_auctoencoder_metrics(history_auctoencoder, autoencoder_weights)

        return autoencoder_model

    @staticmethod
    def plot_auctoencoder_metrics(history_auctoencoder, autoencoder_weights):
        loss_train = history_auctoencoder.history['loss']
        loss_val = history_auctoencoder.history['val_loss']
        plt.plot(np.arange(1,n_epochs+1), loss_train, 'r', label='Training loss')
        plt.plot(np.arange(1,n_epochs+1), loss_val, 'b', label='validation loss')
        plt.title('Training and Validation loss for {}'.format(autoencoder_weights.split('/')[1][:-3]))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(autoencoder_loss_img.format(autoencoder_weights.split('/')[1][:-3]))
        plt.legend()
        plt.show()

    @staticmethod
    def encoder(input_dim, autoencoder_model):
        inputs = Input(shape=(2 * input_dim,))
        layer1 = Dense(dim1, activation='relu', trainable=False, weights=autoencoder_model.layers[1].get_weights())(inputs)
        x = Dense(dim1, activation='relu', trainable=False, weights=autoencoder_model.layers[2].get_weights())(layer1)
        x = Dense(dim2, activation='relu', trainable=False, weights=autoencoder_model.layers[3].get_weights())(x)
        x = Dense(dim2, activation='relu', trainable=False, weights=autoencoder_model.layers[4].get_weights())(x)
        layer3 = Dense(dim3, activation='relu', trainable=False, weights=autoencoder_model.layers[5].get_weights())(x)

        return inputs, layer3

    def train_autoencoder(self):
        ssp_autoencoder = DDImodel.autoencoder(self.Xssp, n_ssp, ssp_weights)
        tsp_autoencoder = DDImodel.autoencoder(self.Xtsp, n_tsp, tsp_weights)
        gsp_autoencoder = DDImodel.autoencoder(self.Xgsp, n_gsp, gsp_weights)

        return ssp_autoencoder, tsp_autoencoder, gsp_autoencoder

    @staticmethod
    def get_encoder_output(ssp_autoencoder, tsp_autoencoder, gsp_autoencoder):
        ssp_input, ssp_out = DDImodel.encoder(n_ssp, ssp_autoencoder)
        tsp_input, tsp_out = DDImodel.encoder(n_tsp, tsp_autoencoder)
        gsp_input, gsp_out = DDImodel.encoder(n_gsp, gsp_autoencoder)

        inputs = ssp_input, tsp_input, gsp_input
        outputs = ssp_out, tsp_out, gsp_out
        return inputs, outputs

    def build_dnn_input(self):
        ssp_autoencoder, tsp_autoencoder, gsp_autoencoder = self.train_autoencoder()
        inputs, outputs = DDImodel.get_encoder_output(ssp_autoencoder, tsp_autoencoder, gsp_autoencoder)

        ssp_out, tsp_out, gsp_out = outputs

        merged = concatenate([ssp_out, tsp_out])
        dnn_input = concatenate([merged,  gsp_out])
        return inputs, dnn_input

    def dnn(self):
        if os.path.exists(dnn_weights):
            print("\n Final Model Loading !!!")
            dnn_model = load_model(dnn_weights)
            dnn_model.compile(
                            optimizer='adam',
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy']
                            )
        else:
            inputs, dnn_input = self.build_dnn_input()

            ssp_input, tsp_input, gsp_input = inputs

            print("\n Final Model Training !!!")
            x = Dense(dense1, activation='relu')(dnn_input)
            x = Dense(dense2, activation='relu')(x)
            x = Dense(dense3, activation='relu')(x)
            output = Dense(dense4, activation='softmax')(x)

            dnn_model = Model(inputs = [ssp_input, tsp_input, gsp_input],
                              outputs= output,
                              name='Final DNN')

            dnn_model.summary()

            dnn_model.compile(
                            optimizer= Adam(learning_rate),
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy']
                            )

            self.history_dnn = dnn_model.fit([self.Xssp, self.Xtsp, self.Xgsp],
                                             self.Y,
                                             epochs=dnn_epoches,
                                             batch_size=batch_size,
                                             validation_split=val_split
                                             )
            dnn_model.save(dnn_weights)

            self.plot_dnn_metrics()
        self.dnn_model = dnn_model

    def plot_dnn_metrics(self):

        loss_train = self.history_dnn.history['loss']
        loss_val = self.history_dnn.history['val_loss']
        plt.plot(np.arange(1,dnn_epoches+1), loss_train, 'r', label='Training loss')
        plt.plot(np.arange(1,dnn_epoches+1), loss_val, 'b', label='validation loss')
        plt.title('Training and Validation loss for Final Model')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(dnn_loss_img)
        plt.legend()
        plt.show()

        acc_train = self.history_dnn.history['accuracy']
        acc_val = self.history_dnn.history['val_accuracy']
        plt.plot(np.arange(1,dnn_epoches+1), acc_train, 'r', label='Training Accuracy')
        plt.plot(np.arange(1,dnn_epoches+1), acc_val, 'b', label='validation Accuracy')
        plt.title('Training and Validation Accuracy for Final Model')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.savefig(dnn_acc_img)
        plt.legend()
        plt.show()

    def predictions(self, A_drug, B_drug):
        X = get_prediction_data(A_drug, B_drug)
        P = self.dnn_model.predict(X).squeeze().argmax(axis=-1)
        return P

    def update_db(self, A_drug, B_drug):
        engine = create_engine(db_url)
        if table_name in sqlalchemy.inspect(engine).get_table_names():
            data = pd.read_sql_table(table_name, db_url)
            df_length = len(data.values)
            prediction = self.predictions(A_drug, B_drug)
            data.loc[df_length+1] = [A_drug, B_drug, int(prediction)]
            with engine.connect() as conn, conn.begin():
                data.to_sql(table_name, conn, if_exists='append', index=False)
        else:
            print("Create a Table named {}".format(table_name))