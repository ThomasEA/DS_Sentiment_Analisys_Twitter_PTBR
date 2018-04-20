# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 23:26:50 2018

@author: evert

Análise de sentimento sobre tweets em PT-BR, classificados como negativo, neutro e positivo
"""

import nltk
import re
import random
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from nltk.corpus import stopwords

##Faz o download das stopwords
#nltk.download('stopwords')

#carrega o dataset de tweets
df = pd.read_csv('./input/Tweets_Mg.csv', encoding='utf-8')

##imprime os primeiros 5 registros
#print(dataset.head) 

##imprime sumarização das linhas e colunas
#print(dataset.count())

##imprime o primeiro registro
#print(dataset.iloc[0])

##imprime os registro 0, 10 e 20
#print(dataset.iloc[[0,10,20]]['Classificacao'])

##imprime as colunas do dataset
#print(dataset.columns)

##imprime os últimos 5 registros
#print(dataset.tail())

##importante: apresenta uma análise estatística para cada coluna do dataset
#print(dataset.describe())

##filtra o dataset por uma determinada coluna
#print(dataset[dataset.Classificacao == 'Positivo'].count())

######################################################################################
#faz um shuffle dos dados
#
#   IMPORTANTE: SÓ COM O SHUFFLE GANHA 5% DE ACURÁCIA E RECALL/PRECISION VÃO A 0.96
######################################################################################
dataset = df.sample(frac=1).reset_index(drop=True)

stops = set(stopwords.words("portuguese"))

######################################
# Separando os dados em suas classes
######################################
tweets = dataset["Text"].values

classificacao = dataset["Classificacao"].values

######################################
# Abordagem Bag of Words
# Cria vetores com a frequencia de ocorrencia das palavras
# Ex.:
# ["olá", "sim", "posso", "não"]
# [1,8,2,6] -> frequencia com que as palavras ocorrem e sua relação com a classificação (neutro, positivo, negativo)
######################################

vectorizer = CountVectorizer(analyzer = "word", ngram_range=(1,2), stop_words = stops, lowercase=False)#, min_df = 1)

#    lowercase (default True) convert all text to lowercase before tokenizing
#    min_df (default 1) remove terms from the vocabulary that occur in fewer than min_df documents (in a large corpus this may be set to 15 or higher to eliminate very rare words)
#    vocabulary ignore words that do not appear in the provided list of words
#    strip_accents remove accents
#    token_pattern (default u'(?u)\b\w\w+\b') regular expression identifying tokens–by default words that consist of a single character (e.g., ‘a’, ‘2’) are ignored, setting token_pattern to '(?u)\b\w+\b' will include these tokens
#    tokenizer (default unused) use a custom function for tokenizing


freq_tweets = vectorizer.fit_transform(tweets)

#Aplica o algoritmo Naive Bayes para treinar sobre os dados
modelo = MultinomialNB()
modelo.fit(freq_tweets, classificacao)

resultados = cross_val_predict(modelo, freq_tweets, classificacao, cv = 10)

###########################################
#imprime os resultados
###########################################
sentimentos = ["Positivo", "Neutro", "Negativo"]

# Lembrando que:
#    : precision = true positive / (true positive + false positive)
#    : recall    = true positive / (true positive + false negative)
#    : f1-score  = 2 * ((precision * recall) / (precision + recall))

print(metrics.classification_report(classificacao, resultados, sentimentos))
print('Accuracy: ' + str(metrics.accuracy_score(classificacao, resultados)))
print(pd.crosstab(classificacao, resultados, rownames = ["Real"], colnames = ["Predito"], margins = True))
