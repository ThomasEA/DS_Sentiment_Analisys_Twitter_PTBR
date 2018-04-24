# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 21:25:47 2018

@author: evert

Cross-validation com Naive Bayes, Decision Tree e SVM

"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#Importando e configurando classificador (SVM)
from sklearn.svm import LinearSVC
#Importando e configurando classificador (Naive Bayes)
from sklearn.naive_bayes import MultinomialNB
#Importando e configurando classificador (DecisionTree)
from sklearn.tree import DecisionTreeClassifier
#Importando gerador de parametros otimizados para SVM e Decicion Tree
from sklearn.model_selection import GridSearchCV

sentimentos = ["Positivo", "Neutro", "Negativo"]

def cross_validate(model, X, labels, vectorizer, n_folds = 10):
    """
    Executa o processo de Cross Validation utilizando o algoritmo de predição
    passado como parâmetro
    
    Keyword arguments:
        model -- Algoritmo de predição (ex. Naive Bayes, Decision Tree, SVM, ...)
        
        X -- dataset com colunas a serem analisadas
        
        labels -- rótulos para o dataset
        
        vectorizer -- NLTK Vectorizer
        
        n_folds -- número de folds para o Cross Validation
        
    Return:
        totalMat -- Matriz de confusão (3x3)
        
        total -- Acurácia
        
        f1_score -- f1 Score
        
        precision -- precision score
        
        recall -- recall score
    """
    kf = StratifiedKFold(n_splits=n_folds)
    
    total = 0
    f1score = 0
    recall = 0
    precision = 0
    totalMat = np.zeros((3,3))
    
    cf_matrix_real = []
    cf_matrix_predict = []
    
    for train_index, test_index in kf.split(X,labels):
        X_train = [X[i] for i in train_index]
        X_test = [X[i] for i in test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        train_corpus = vectorizer.fit_transform(X_train) 
        test_corpus = vectorizer.transform(X_test)
        
        cf_matrix_real = np.concatenate([cf_matrix_real, y_test])
        
        model.fit(train_corpus,y_train)
        result = model.predict(test_corpus)
        
        cf_matrix_predict = np.concatenate([cf_matrix_predict, result])
        
        totalMat = totalMat + confusion_matrix(y_test, result)
        #total = total+sum(y_test==result)
        total = total + metrics.accuracy_score(y_test, result)

    total               = total / n_folds * 100
    f1score             = metrics.f1_score(cf_matrix_real, cf_matrix_predict, sentimentos, average='weighted')
    recall              = metrics.recall_score(cf_matrix_real, cf_matrix_predict, sentimentos, average='weighted')
    precision           = metrics.precision_score(cf_matrix_real, cf_matrix_predict, sentimentos, average='weighted')
    # Lembrando que:
    #    : precision = true positive / (true positive + false positive)
    #    : recall    = true positive / (true positive + false negative)
    #    : f1-score  = 2 * ((precision * recall) / (precision + recall))
    print("##################################")
    print(metrics.classification_report(cf_matrix_real, cf_matrix_predict, sentimentos))
    print('Accuracy: ' + str(total))
    print(pd.crosstab(cf_matrix_real, cf_matrix_predict, rownames = ["Real"], colnames = ["Predito"], margins = True))
    print("##################################")
    
    #print(pd.crosstab(y_train, result, rownames = ["Real"], colnames = ["Predito"], margins = True))    

    return totalMat, total, f1score, precision, recall

def apply_model(model, X, labels, vectorizer):
    """
    Aplica o modelo previamente treinado em um conjunto de dados
    
    Keyword arguments:
        model -- Algoritmo de predição (ex. Naive Bayes, Decision Tree, SVM, ...)
        
        X -- dataset com colunas a serem analisadas
        
        labels -- rótulos para o dataset
        
        vectorizer -- NLTK Vectorizer
        
    Return:
        totalMat -- Matriz de confusão (3x3)
        
        total -- Acurácia
        
        f1_score -- f1 Score
        
        precision -- precision score
        
        recall -- recall score
    """
    test_corpus = vectorizer.transform(X)
        
    result = model.predict(test_corpus)
    
    total               = metrics.accuracy_score(labels, result)
    f1score             = metrics.f1_score(labels, result, sentimentos, average='weighted')
    recall              = metrics.recall_score(labels, result, sentimentos, average='weighted')
    precision           = metrics.precision_score(labels, result, sentimentos, average='weighted')
    # Lembrando que:
    #    : precision = true positive / (true positive + false positive)
    #    : recall    = true positive / (true positive + false negative)
    #    : f1-score  = 2 * ((precision * recall) / (precision + recall))
    print(">>>>>>>>>>>>> TEST MODEL <<<<<<<<<<<<<<")
    print(metrics.classification_report(labels, result, sentimentos))
    print('Accuracy: ' + str(total))
    print(pd.crosstab(labels, result, rownames = ["Real"], colnames = ["Predito"], margins = True))
    print(">>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<")
    
    #print(pd.crosstab(y_train, result, rownames = ["Real"], colnames = ["Predito"], margins = True))    


def process_data_bag_of_words(X_train, y_train, X_test, y_test, stopwords = None, n_folds = 10, n_grams = 1):
    result = pd.DataFrame()
    
    """
    Processa os dados de treino e teste
    """
    #vectorizer = CountVectorizer(analyzer = "word", ngram_range=(1,n_grams), stop_words = stopwords)#, lowercase=True)#, min_df = 1)
    vectorizer = TfidfVectorizer(sublinear_tf=True, use_idf=True,stop_words=stopwords, ngram_range=(1,n_grams))#, lowercase=True, )
    
    nb = MultinomialNB()
    matrix_conf_nb, accur_nb, f1_nb, precision_nb, recall_nb        = cross_validate(nb, X_train, y_train, vectorizer)

    apply_model(nb, X_test, y_test, vectorizer)

    #parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
    #    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    svm = LinearSVC()
    #svm = GridSearchCV(svm, parameters)    
    matrix_conf_svm, accur_svm, f1_svm, precision_svm, recall_svm   = cross_validate(svm, X_train, y_train, vectorizer)
    
    apply_model(svm, X_test, y_test, vectorizer)
    
    clf_param = {'max_depth': range(3,10)}
    dt = DecisionTreeClassifier(max_depth=10)
    dt = GridSearchCV(dt, clf_param)
    matrix_conf_dt, accur_dt, f1_dt, precision_dt, recall_dt        = cross_validate(dt, X_train, y_train, vectorizer)
    
    apply_model(dt, X_test, y_test, vectorizer)
    
    n_grams_str = ''
    
    if (n_grams == 1):
        n_grams_str = 'unigram'
    elif (n_grams == 2):
        n_grams_str = 'bigram'
    elif (n_grams == 3):
        n_grams_str = 'trigram'
    
    #Sumariza os resultados por classificador
    result.loc[1, 'Classificador']      = 'Naive Bayes'
    result.loc[1, 'Acurácia']           = accur_nb
    result.loc[1, 'F1Score']            = f1_nb
    result.loc[1, 'NGrams']             = n_grams_str
    result.loc[1, 'Recall']             = recall_nb
    result.loc[1, 'Precision']          = precision_nb
    

    result.loc[2, 'Classificador']      = 'SVM'
    result.loc[2, 'Acurácia']           = accur_svm
    result.loc[2, 'F1Score']            = f1_svm
    result.loc[2, 'NGrams']             = n_grams_str
    result.loc[2, 'Recall']             = recall_svm
    result.loc[2, 'Precision']          = precision_svm
    
    result.loc[3, 'Classificador']      = 'Decision Tree'
    result.loc[3, 'Acurácia']           = accur_dt
    result.loc[3, 'F1Score']            = f1_dt
    result.loc[3, 'NGrams']             = n_grams_str
    result.loc[3, 'Recall']             = recall_dt
    result.loc[3, 'Precision']          = precision_dt

    return result