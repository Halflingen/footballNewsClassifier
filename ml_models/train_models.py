import numpy as np
import NaiveBayes
import LinearSVC
import SVC

def train_nb():
    nb = NaiveBayes.NaiveBayes()
    vectorizer = ['tfidf']
    dist_type = ['bernoulli']
    alpha = [i for i in np.arange(0.03, 4, 0.03)]
    parameters = \
            (vectorizer,
            dist_type,
            alpha)

    #nb.find_optimal_parameters(parameters)
    print("1 nb")

    vectorizer = ['count']
    dist_type = ['bernoulli']
    alpha = [i for i in np.arange(0.03, 4, 0.03)]
    parameters = \
            (vectorizer,
            dist_type,
            alpha)

    #nb.find_optimal_parameters(parameters)
    print("2 nb")

    vectorizer = ['tfidf']
    dist_type = ['multinomial']
    alpha = [i for i in np.arange(0.01, 4, 0.03)]
    parameters = \
            (vectorizer,
            dist_type,
            alpha)

    #nb.find_optimal_parameters(parameters)
    print("3 nb")

    vectorizer = ['count']
    dist_type = ['multinomial']
    alpha = [i for i in np.arange(0.01, 4, 0.03)]
    parameters = \
            (vectorizer,
            dist_type,
            alpha)

    #nb.find_optimal_parameters(parameters)
    print("4 nb")

def train_LinearSVC():
    linear_svc = LinearSVC.LinearSVC_()
    vectorizer = ['tfidf']
    c1 = [i for i in np.arange(0.005, 0.25, 0.008)]
    c2 = [i for i in np.arange(0.25, 1.5, 0.02)]
    c = c1+c2
    tol = [1e-4]
    penalty = ['l2']
    loss = ['squared_hinge']
    dual = [True]
    multi_class = ['ovr']
    fit_intercept = [True]
    intercept_scaling = [2]
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
    print("0 linearSVC")

    linear_svc = LinearSVC.LinearSVC_()
    vectorizer = ['tfidf']
    c = [0.0434]
    tol = [1e-4]
    penalty = ['l2']
    loss = ['squared_hinge']
    dual = [True]
    multi_class = ['ovr']
    fit_intercept = [True]
    intercept_scaling = [i for i in np.arange(1, 10, 0.2)]
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
    print("1 linearSVC")

    linear_svc = LinearSVC.LinearSVC_()
    vectorizer = ['tfidf']
    c = [0.0434]
    tol = [i for i in np.arange(0.00001, 0.001, 0.00001)]
    penalty = ['l2']
    loss = ['squared_hinge']
    dual = [True]
    multi_class = ['ovr']
    fit_intercept = [True]
    intercept_scaling = [2]
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
    print("2 linearSVC")


#chanched c value from 1 to 3.5 and gamma from 0.75 to 1
def train_SVC_rbf():
    svc = SVC.SVC_()
    vectorizer = ['tfidf']
    c = [i for i in np.arange(0.05,2.5,0.03)]
    tol = [1e-3]
    gamma = [0.5]
    kernel = ['rbf']
    degree = [9]
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

    #svc.find_optimal_parameters(parameters)
    print('1 SVC rbf')

    vectorizer = ['tfidf']
    #c = [i for i in np.arange(0.25,1,0.25)]
    c = [2.0]
    tol = [1e-3]
    gamma = [i for i in np.arange(0.05,3,0.035)]
    #gamma = [0.75]
    kernel = ['rbf']
    degree = [9]
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
    print('2 SVC rbf')

    vectorizer = ['tfidf']
    c = [i for i in np.arange(0.05,4,0.05)]
    tol = [1e-3]
    gamma = [0.785]
    kernel = ['rbf']
    degree = [5]
    coef0 = [0.0]
    shrinking = [False]
    parameters = \
            (vectorizer,
            c,
            tol,
            gamma,
            kernel,
            degree,
            coef0,
            shrinking)

    #svc.find_optimal_parameters(parameters)
    print('3 SVC rbf')

    vectorizer = ['tfidf']
    #c = [i for i in np.arange(0.25,1,0.25)]
    c = [2.6]
    tol = [1e-3]
    gamma = [i for i in np.arange(0.05,4,0.05)]
    #gamma = [0.75]
    kernel = ['rbf']
    degree = [5]
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

    #svc.find_optimal_parameters(parameters)
    print('4 SVC rbf')


def train_SVC_poly():
    svc = SVC.SVC_()
    vectorizer = ['tfidf']
    c = [i for i in np.arange(0.5,1.5,0.0125)]
    #c = [i for i in np.arange(0.07,0.4,0.02)]
    tol = [1e-3]
    gamma = [1]
    kernel = ['poly']
    degree = [2]
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

    #svc.find_optimal_parameters(parameters)
    print('1 SVC poly')

    vectorizer = ['tfidf']
    c = [0.96]
    tol = [1e-3]
    #gamma = [i for i in np.arange(0.50,2.5,0.02)]
    gamma = [i for i in np.arange(0.5,1.5,0.0125)]
    #gamma = [i for i in np.arange(0.5,0.8,0.02)]
    kernel = ['poly']
    degree = [2]
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

    #svc.find_optimal_parameters(parameters)
    print('2 SVC poly')

    vectorizer = ['tfidf']
    c = [0.96]
    tol = [1e-3]
    gamma = [1.2]
    kernel = ['poly']
    degree = [2]
    coef0 = [i for i in np.arange(0.00,3,0.0375)]
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
    print('3 SVC poly')

    vectorizer = ['tfidf']
    c = [i for i in np.arange(0.05,3,0.05)]
    #c = [i for i in np.arange(0.07,0.4,0.02)]
    tol = [1e-3]
    gamma = [2.355]
    kernel = ['poly']
    degree = [2]
    coef0 = [1.5001]
    shrinking = [False]
    parameters = \
            (vectorizer,
            c,
            tol,
            gamma,
            kernel,
            degree,
            coef0,
            shrinking)

    #svc.find_optimal_parameters(parameters)
    print('4 SVC poly')

    vectorizer = ['tfidf']
    c = [1.59]
    tol = [1e-3]
    gamma = [i for i in np.arange(0.05,3,0.05)]
    #gamma = [i for i in np.arange(0.5,0.8,0.02)]
    kernel = ['poly']
    degree = [2]
    coef0 = [1.5001]
    shrinking = [False]
    parameters = \
            (vectorizer,
            c,
            tol,
            gamma,
            kernel,
            degree,
            coef0,
            shrinking)

    #svc.find_optimal_parameters(parameters)
    print('5 SVC poly')

    vectorizer = ['tfidf']
    c = [1.59]
    tol = [1e-3]
    gamma = [2.355]
    kernel = ['poly']
    degree = [2]
    coef0 = [i for i in np.arange(0.05,4,0.05)]
    shrinking = [False]
    parameters = \
            (vectorizer,
            c,
            tol,
            gamma,
            kernel,
            degree,
            coef0,
            shrinking)

    #svc.find_optimal_parameters(parameters)
    print('6 SVC poly')


def main():
    #train_nb()
    print('nb done')
    #train_LinearSVC()
    print('LinearSVC done')
    #train_SVC_rbf()
    print('SVC RBF done')
    train_SVC_poly()
    print('SVC Poly  done')

main()
