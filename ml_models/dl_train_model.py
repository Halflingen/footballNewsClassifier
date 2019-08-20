import numpy as np
from keras import backend as K
import cnn
import rnn

def train_cnn():
####################################################################
#                            Plot 1                                #
####################################################################
    cnn_ = cnn.CNN()
    embedding_dimension = [i for i in range(10, 250, 3)]
    filter_size = [128]
    kernel_size = [2]
    pool_size = [4]
    dropout_rate = [0.5]
    optimizer = ['adam']
    epochs = [10]
    batch_size = [32]
    dense_layer = [25]
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

    #cnn_.find_optimal_parameters(parameters)
    print("1 done cnn")

####################################################################
#                            Plot 2                                #
####################################################################
    embedding_dimension = [i for i in range(10, 250, 3)]
    filter_size = [64]
    kernel_size = [2]
    pool_size = [4]
    dropout_rate = [0.5]
    optimizer = ['adam']
    epochs = [10]
    batch_size = [32]
    dense_layer = [25]
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

    #cnn_.find_optimal_parameters(parameters)
    print("2 done cnn")

####################################################################
#                            Plot 3                                #
####################################################################
    embedding_dimension = [i for i in range(10, 250, 3)]
    filter_size = [32]
    kernel_size = [2]
    pool_size = [4]
    dropout_rate = [0.5]
    optimizer = ['adam']
    epochs = [10]
    batch_size = [32]
    dense_layer = [25]
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

    #cnn_.find_optimal_parameters(parameters)
    print("3 done cnn")


####################################################################
#                            Plot 5                                #
####################################################################
    embedding_dimension = [i for i in range(10, 250, 3)]
    filter_size = [256]
    kernel_size = [2]
    pool_size = [4]
    dropout_rate = [0.5]
    optimizer = ['adam']
    epochs = [10]
    batch_size = [32]
    dense_layer = [25]
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

    #cnn_.find_optimal_parameters(parameters)
    print("5 done cnn")


####################################################################
#                            Plot 6                                #
####################################################################
    embedding_dimension = [i for i in range(10, 250, 3)]
    filter_size = [128]
    kernel_size = [3]
    pool_size = [2]
    dropout_rate = [0.5]
    optimizer = ['adam']
    epochs = [10]
    batch_size = [32]
    dense_layer = [25]
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

    #cnn_.find_optimal_parameters(parameters)
    print("6 done cnn")

####################################################################
#                            Plot 7                                #
####################################################################
    embedding_dimension = [i for i in range(10, 250, 3)]
    filter_size = [64]
    kernel_size = [3]
    pool_size = [2]
    dropout_rate = [0.5]
    optimizer = ['adam']
    epochs = [10]
    batch_size = [32]
    dense_layer = [25]
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

    #cnn_.find_optimal_parameters(parameters)
    print("7 done cnn")

####################################################################
#                            Plot 8                                #
####################################################################
    embedding_dimension = [i for i in range(10, 250, 3)]
    filter_size = [32]
    kernel_size = [3]
    pool_size = [2]
    dropout_rate = [0.5]
    optimizer = ['adam']
    epochs = [10]
    batch_size = [32]
    dense_layer = [25]
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

    #cnn_.find_optimal_parameters(parameters)
    print("8 done cnn")


####################################################################
#                            Plot 10                               #
####################################################################
    embedding_dimension = [i for i in range(10, 250, 3)]
    filter_size = [256]
    kernel_size = [3]
    pool_size = [2]
    dropout_rate = [0.5]
    optimizer = ['adam']
    epochs = [10]
    batch_size = [32]
    dense_layer = [25]
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

    #cnn_.find_optimal_parameters(parameters)
    print("10 done cnn")

####################################################################
#                            Plot 6                                #
####################################################################
    embedding_dimension = [i for i in range(10, 250, 3)]
    filter_size = [128]
    kernel_size = [6]
    pool_size = [2]
    dropout_rate = [0.5]
    optimizer = ['adam']
    epochs = [10]
    batch_size = [32]
    dense_layer = [25]
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

    #cnn_.find_optimal_parameters(parameters)
    print("6 done cnn")

####################################################################
#                            Plot 7                                #
####################################################################
    embedding_dimension = [i for i in range(10, 250, 3)]
    filter_size = [64]
    kernel_size = [6]
    pool_size = [2]
    dropout_rate = [0.5]
    optimizer = ['adam']
    epochs = [10]
    batch_size = [32]
    dense_layer = [25]
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

    #cnn_.find_optimal_parameters(parameters)
    print("7 done cnn")

####################################################################
#                            Plot 8                                #
####################################################################
    embedding_dimension = [i for i in range(10, 250, 3)]
    filter_size = [32]
    kernel_size = [6]
    pool_size = [2]
    dropout_rate = [0.5]
    optimizer = ['adam']
    epochs = [10]
    batch_size = [32]
    dense_layer = [25]
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

    #cnn_.find_optimal_parameters(parameters)
    print("8 done cnn")


####################################################################
#                            Plot 10                               #
####################################################################
    embedding_dimension = [i for i in range(10, 250, 3)]
    filter_size = [256]
    kernel_size = [6]
    pool_size = [2]
    dropout_rate = [0.5]
    optimizer = ['adam']
    epochs = [10]
    batch_size = [32]
    dense_layer = [25]
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

    #cnn_.find_optimal_parameters(parameters)
    print("10 done cnn")


####################################################################
#                            Plot 3                                #
####################################################################
    embedding_dimension = [210]
    filter_size = [i for i in range(10,256,3)]
    kernel_size = [6]
    pool_size = [2]
    dropout_rate = [0.5]
    optimizer = ['adam']
    epochs = [10]
    batch_size = [32]
    dense_layer = [25]
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

    #cnn_.find_optimal_parameters(parameters)
    print("15 done cnn")


####################################################################
#                            Plot 3                                #
####################################################################
    embedding_dimension = [210]
    filter_size = [182]
    kernel_size = [6]
    pool_size = [2]
    dropout_rate = [0.5]
    optimizer = ['adam']
    epochs = [10]
    batch_size = [32]
    dense_layer = [i for i in range(10, 250, 3)]
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

    #cnn_.find_optimal_parameters(parameters)
    print("16 done cnn")

####################################################################
#                            Plot 3                                #
####################################################################
    embedding_dimension = [210]
    filter_size = [182]
    kernel_size = [6]
    pool_size = [2]
    dropout_rate = [i for i in np.arange(0.01, 1.0, 0.0125)]
    optimizer = ['adam']
    epochs = [10]
    batch_size = [32]
    dense_layer = [178]
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

    #cnn_.find_optimal_parameters(parameters)
    print("11 done cnn")


####################################################################
#                                                                  #
#                            RNN                                   #
#                                                                  #
####################################################################
def train_rnn():
####################################################################
#                            Plot 1                                #
####################################################################
    rnn_ = rnn.RNN()
    embedding_dimension = [i for i in range(10, 250, 3)]
    LSTM_neurons = [128]
    dropout_rate = [0.5]
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

    #rnn_.find_optimal_parameters(parameters)
    print("1 done rnn")




####################################################################
#                            Plot 2                                #
####################################################################
    embedding_dimension = [i for i in range(10, 250, 3)]
    LSTM_neurons = [64]
    dropout_rate = [0.5]
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

    #rnn_.find_optimal_parameters(parameters)
    print("2 done rnn")

####################################################################
#                            Plot 3                                #
####################################################################
    embedding_dimension = [i for i in range(10, 250, 3)]
    LSTM_neurons = [32]
    dropout_rate = [0.5]
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

    #rnn_.find_optimal_parameters(parameters)
    print("3 done rnn")


####################################################################
#                            Plot 5                                #
####################################################################
    embedding_dimension = [i for i in range(10, 250, 3)]
    LSTM_neurons = [256]
    dropout_rate = [0.5]
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

    #rnn_.find_optimal_parameters(parameters)
    print("5 done rnn")

####################################################################
#                            Plot 6                                #
####################################################################
    embedding_dimension = [125]
    LSTM_neurons = [105]
    dropout_rate = [i for i in np.arange(0.01, 1.0, 0.0125)]
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

    #rnn_.find_optimal_parameters(parameters)
    print("6 done rnn")

####################################################################
#                            Plot 7                                #
####################################################################
    embedding_dimension = [125]
    LSTM_neurons = [105]
    dropout_rate = [i for i in np.arange(0.01, 1.0, 0.0125)]
    dropout_location = ['after']
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

    #rnn_.find_optimal_parameters(parameters)
    print("7 done rnn")

####################################################################
#                            Plot 8                                #
####################################################################
    embedding_dimension = [125]
    LSTM_neurons = [105]
    dropout_rate = [i for i in np.arange(0.01, 1.0, 0.0125)]
    dropout_location = ['none']
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

    #rnn_.find_optimal_parameters(parameters)
    print("8 done rnn")

####################################################################
#                            Plot 8                                #
####################################################################
    embedding_dimension = [125]
    LSTM_neurons = [i for i in range(10,256,3)]
    dropout_rate = [0.5]
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

    #rnn_.find_optimal_parameters(parameters)
    print("9 done rnn")

def main():
    #train_cnn()
    print('cnn done')
    #train_rnn()
    print('rnn done')


if __name__ == '__main__':
    main()
