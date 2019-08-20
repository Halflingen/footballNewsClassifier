#SVC doesent work on the dataset. There are to many samples. LinearSVC works
#and perfomrs slighly better than naive bayes
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
import time
import numpy as np
import pickle

import get_data
import metrics


class SVC_:
    def __init__(self):
        self.data = get_data.Data()
        self.train_data,\
        self.train_target,\
        self.test_data,\
        self.test_target\
        = self.data.fetch_train_data()


    def create_model(self, para):
        '''
        0 - vectorizer
        1 - c
        2 - tol
        3 - gamma
        4 - kernel
        5 - degree
        6 - coef0
        7 - shrinking
        '''
        vectorizer = TfidfVectorizer()
        if para[0] == 'countVec':
            vectorizer = CountVectorizer()

        clf = SVC(C=para[1], gamma=para[3], kernel=para[4], coef0=para[6],
                degree=para[5], shrinking=para[7], tol=para[2])
        pipeline = Pipeline([
            ('vec', vectorizer),
            ('clf', clf),
        ])

        return pipeline


    def train_model(self, parameters):
        '''
        0 - vectorizer
        1 - c
        2 - tol
        3 - gamma
        4 - kernel
        5 - degree
        6 - coef0
        7 - shrinking
        '''

        metric = metrics.MetricsAndParameters('svc', parameters)

        #if(metric.duplicate()):
            #return

        skf = StratifiedKFold(n_splits=10, shuffle=True)
        for train_index, test_index in skf.split(self.train_data, self.train_target):
            #Data
            train_data = self.train_data[train_index]
            train_target = self.train_target[train_index]
            test_data = self.train_data[test_index]
            test_target = self.train_target[test_index]

            #Training
            train_time = time.time()
            model = self.create_model(parameters)
            model.fit(train_data, train_target)
            train_time =  time.time() - train_time

            #Prediction
            prediction_time = time.time()
            test_results = model.predict(test_data)
            prediction_time =  time.time() - prediction_time

            #Metric calculation
            metric.calculate_metrics(test_target, test_results, train_time, prediction_time)

        #save to database
        metric.store_metrics()
        metric.store_parameters()


    def find_optimal_parameters(self, parameters):
        for a in parameters[0]:
            for b in parameters[1]:
                for c in parameters[2]:
                    for d in parameters[3]:
                        for e in parameters[4]:
                            for f in parameters[5]:
                                for g in parameters[6]:
                                    for h in parameters[7]:
                                        self.train_model((a,b,c,d,e,f,g,h))


    def create_and_save_model(self, parameters):
        train_data = self.train_data
        train_target = self.train_target

        model = self.create_model(parameters)
        model.fit(train_data, train_target)

        name = "t2_model_svc_" + parameters[4] + ".pickle"
        with open(name, 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    svc = SVC_()
    vectorizer = ['tfidf']
    c = [2]
    #c = [1]
    tol = [1e-3]
    #gamma = [i for i in np.arange(0.25,1,0.25)]
    gamma = [0.55]
    kernel = ['rbf']
    degree = [33]
    coef0 = [0.0]
    shrinking = [True]
    parameters = \
            (vectorizer,
            c,
            tol,
            gamma,
            kernel,
            degree,
            coef0,
            shrinking)

    svc.find_optimal_parameters(parameters)
    #para = ('tfidf', 2, 1e-3, 0.55, 'rbf', 2, 0, False)
    #svc.create_and_save_model(para)

    #para = ('tfidf', 0.96, 1e-3, 1.2, 'poly', 2, 0.63, False)
    #svc.create_and_save_model(para)


if __name__ == '__main__':
    main()
