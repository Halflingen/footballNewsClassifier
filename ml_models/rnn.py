#A RNN made with Keras
from keras.layers import Input, Dense, Flatten, Dropout, Embedding, LSTM, CuDNNLSTM, Bidirectional
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.utils.vis_utils import plot_model
from keras.models import Model, load_model, Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import StratifiedKFold
from keras.utils.np_utils import to_categorical
import time
import pickle
import gc
from keras import backend as K

import get_data
import metrics


class RNN:
    def __init__(self):
        self.data = get_data.Data()
        self.train_data,\
        self.train_target,\
        self.test_data,\
        self.test_target\
        = self.data.fetch_train_data()

        self.tokenizer = self.create_tokenizer()
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.word_len = max([len(s.split()) for s in self.train_data])

        self.encoded_train = self.tokenizer.texts_to_sequences(self.train_data)
        self.padded_train = pad_sequences(self.encoded_train, maxlen=self.word_len, padding='post')
        #self.encoded_test = self.tokenizer.texts_to_sequences(self.test_data)
        #self.padded_test = pad_sequences(self.encoded_test, maxlen=self.word_len, padding='post')


    def create_tokenizer(self):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.train_data)
        return tokenizer


    def create_model(self, para):
        '''
        0 - embedding_dimension
        1 - LSTM_neurons
        2 - dropout_rate
        3 - dropout_location
        4 - optimizer
        5 - epochs
        6 - batch_size
        '''
        model = Sequential()
        model.add(Embedding(self.vocab_size, para[0]))
        if para[3] == 'before':
            model.add(Dropout(para[2]))
        model.add(Bidirectional(CuDNNLSTM(units=para[1])))
        if para[3] == 'after':
            model.add(Dropout(para[2]))
        model.add(Dense(5, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=para[4], metrics=['accuracy'])

        #print(model.summary())
        #plot_model(model, show_shapes=True, to_file='my_model_rnn.png')
        return model



    def train_model(self, parameters):
        '''
        0 - embedding_dimension
        1 - LSTM_neurons
        2 - dropout_rate
        3 - dropout_location
        4 - optimizer
        5 - epochs
        6 - batch_size
        '''

        metric = metrics.MetricsAndParameters('rnn', parameters)

        #if(metric.duplicate()):
            #return

        skf = StratifiedKFold(n_splits=10, shuffle=False)
        for train_index, test_index in skf.split(self.padded_train, self.train_target):
            #Data
            train_data = self.padded_train[train_index]
            train_target = to_categorical(self.train_target[train_index],num_classes=5)
            test_data = self.padded_train[test_index]
            test_target = self.train_target[test_index]

            #Training
            train_time = time.time()
            model = self.create_model(parameters)
            model.fit([train_data], train_target, epochs=parameters[5], batch_size=parameters[6])
            train_time =  time.time() - train_time

            #Prediction
            prediction_time = time.time()
            score = model.predict(test_data, batch_size=parameters[6])
            prediction_time =  time.time() - prediction_time

            #Metric calculation
            test_results = score.argmax(axis=-1)
            metric.calculate_metrics(test_target, test_results, train_time, prediction_time)

        #save to database
        metric.store_metrics()
        metric.store_parameters()
        metric = 0
        gc.collect()
        K.clear_session()



    def find_optimal_parameters(self, parameters):
        k = 0
        for a in parameters[0]:
            for b in parameters[1]:
                for c in parameters[2]:
                    for d in parameters[3]:
                        for e in parameters[4]:
                            for f in parameters[5]:
                                for g in parameters[6]:
                                    self.train_model((a,b,c,d,e,f,g))
                                    print(k)
                                    k += 1


    def create_and_save_model(self, parameters):
        train_data = self.padded_train
        train_target = to_categorical(self.train_target,num_classes=5)

        model = self.create_model(parameters)
        model.fit([train_data], train_target, epochs=parameters[5], batch_size=parameters[6])

        model.save('t2_model_rnn.h5')
        with open('t2_tokenizer_rnn.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    rnn = RNN()
    embedding_dimension = [125]
    LSTM_neurons = [105]
    dropout_rate = [0.9]
    dropout_location = ['before']
    optimizer = ['adam']
    epochs = [10]
    batch_size = [32]
    parameters = \
            (embedding_dimension,
            LSTM_neurons,
            dropout_rate,
            dropout_location,
            optimizer,
            epochs,
            batch_size)

    rnn.find_optimal_parameters(parameters)
    #para = (125, 105, 0.9, 'before', 'adam', 10, 32)
    #rnn.create_and_save_model(para)


if __name__ == '__main__':
    main()
