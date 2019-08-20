#A CNN made with Keras
from keras.layers import Input, Dense, Flatten, Dropout, Embedding
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


class CNN:
    def __init__(self):
        self.data = get_data.Data()
        self.train_data,\
        self.train_target,\
        self.test_data,\
        self.test_target\
        = self.data.fetch_train_data()

        self.tokenizer = self.create_tokenizer()
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.word_len = max([len(s.split()) for s in self.train_data]) #change variable name

        self.encoded_train = self.tokenizer.texts_to_sequences(self.train_data)
        self.padded_train = pad_sequences(self.encoded_train, maxlen=self.word_len, padding='post')
        #self.encoded_test = self.tokenizer.texts_to_sequences(self.test_data)
        #self.padded_test = pad_sequences(self.encoded_test, maxlen=self.word_len, padding='post')


    def create_tokenizer(self):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.train_data)
        return tokenizer


    def save_model_and_tokenizer(self, model):
        model.save('model_test.h5')
        with open('tokenizer_cnn.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def create_model(self, para):
        '''
        Parameters:
        0 - emb_dim
        1 - filter_size
        2 - kernel_size
        3 - pool_size
        4 - dropout_rate
        5 - optimizer
        6 - dense_layer
        '''
        model = Sequential()
        model.add(Embedding(input_dim=self.vocab_size, output_dim=para[0], input_length=65))
        model.add(Conv1D(filters=para[1], kernel_size=para[2], activation='relu'))
        model.add(Dropout(rate=para[4]))
        model.add(MaxPooling1D(pool_size=para[3]))
        model.add(Flatten())
        model.add(Dense(units=para[6], activation='relu'))
        model.add(Dense(units=5, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=para[5], metrics=['accuracy'])

        #print(model.summary())
        #plot_model(model, show_shapes=True, to_file='my_model_cnn.png')
        return model


    def train_model(self, parameters):
        '''
        Parameters:
        0 - emb_dim
        1 - filter_size
        2 - kernel_size
        3 - pool_size
        4 - dropout_rate
        5 - optimizer
        6 - dense_layer
        7 - epoch
        8 - batch_size
        '''

        metric = metrics.MetricsAndParameters('cnn', parameters)

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
            model.fit([train_data], train_target, epochs=parameters[7], batch_size=parameters[8], verbose=0)
            train_time =  time.time() - train_time

            #Prediction
            prediction_time = time.time()
            score = model.predict(test_data, batch_size=parameters[8])
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
                                    for h in parameters[7]:
                                        for i in parameters[8]:
                                            self.train_model((a,b,c,d,e,f,g,h,i))
                                            print(k)
                                            k += 1


    def create_and_save_model(self, parameters):
        train_data = self.padded_train
        train_target = to_categorical(self.train_target,num_classes=5)

        model = self.create_model(parameters)
        model.fit([train_data], train_target, epochs=parameters[7], batch_size=parameters[8])

        model.save('t2_model_cnn.h5')
        with open('t2_tokenizer_cnn.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    cnn = CNN()
    embedding_dimension = [400]
    filter_size = [32]
    kernel_size = [2]
    pool_size = [4]
    dropout_rate = [0.1115]
    optimizer = ['adam']
    epochs = [10]
    batch_size = [32]
    dense_layer = [1024]
    parameters = \
            (embedding_dimension,
            filter_size,
            kernel_size,
            pool_size,
            dropout_rate,
            optimizer,
            dense_layer,
            epochs,
            batch_size)

    #cnn.find_optimal_parameters(parameters)
    para = (210, 182, 6, 2, 0.09, 'adam', 178, 10, 32)
    #cnn.create_and_save_model(para)


if __name__ == '__main__':
    main()
