from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
import time
import numpy as np
import pickle

import get_data
import metrics


class LinearSVC_:
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
        3 - penalty
        4 - loss
        5 - dual
        6 - multi_class
        7 - fit_intercept
        8 - intercept_scaling
        '''
        vectorizer = CountVectorizer()
        if vectorizer == 'tfidf':
            vectorizer = TfidfVectorizer()

        clf = LinearSVC(C=para[1], penalty=para[3], tol=para[2], loss=para[4], dual=para[5],
                multi_class=para[6], fit_intercept=para[7], intercept_scaling=para[8], max_iter=5000)
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
        3 - penalty
        4 - loss
        5 - dual
        6 - multi_class
        7 - fit_intercept
        8 - intercept_scaling
        '''

        metric = metrics.MetricsAndParameters('linear_svc', parameters)

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
                                        for i in parameters[8]:
                                            if d == 'l1' and e=='squared_hinge' and f==True:
                                                continue
                                            if d == 'l2' and e=='hinge' and f==False:
                                                continue
                                            if d == 'l1' and e == 'hinge':
                                                continue
                                            self.train_model((a,b,c,d,e,f,g,h,i))


    def create_and_save_model(self, parameters):
        train_data = self.train_data
        train_target = self.train_target

        model = self.create_model(parameters)
        model.fit(train_data, train_target)

        with open('t2_model_linearSVC.pickle', 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    linear_svc = LinearSVC_()
    vectorizer = ['tfidf']
    c = [i for i in np.arange(0.01, 1, 0.01)]
    tol = [1e-4]
    penalty = ['l2']
    loss = ['squared_hinge']
    dual = [True]
    multi_class = ['ovr']
    fit_intercept = [True]
    intercept_scaling = [1.01]
    parameters = \
            (vectorizer,
            c,
            tol,
            penalty,
            loss,
            dual,
            multi_class,
            fit_intercept,
            intercept_scaling)

    #linear_svc.find_optimal_parameters(parameters)
    para = ('tfidf', 0.052, 1e-4, 'l2', 'squared_hinge', True, 'ovr', True, 1.0)
    linear_svc.create_and_save_model(para)


if __name__ == '__main__':
    main()
