# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 23:26:50 2018

@author: evert

Análise de sentimento sobre tweets em PT-BR, classificados como negativo, neutro e positivo
"""

import pandas as pd
import numpy as np

import clean_data as clean
import vis_data as visualization
import process_model as process_model

from sklearn.cross_validation import train_test_split

# Limpa as variáveis a cada execução
def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

clear_all()

generate_visualization = True

#carrega o dataset de tweets
df = pd.read_csv('./input/Tweets_Mg.csv', encoding='utf-8')

visualization.distr_qtd_carac(df)

################################################################
# Faz uma limpeza prévia da coluna de texto do tweet
################################################################
dataset = clean.clean_data(dataset=df, shuffle=False)

dataset, stops = clean.apply_text_processing(dataset)
######################################
# Separando os dados em suas classes
######################################
tweets = dataset["Text"].values
classificacao = dataset["Classificacao"].values

######################################
# Divide o dataset:
#   80% para treino
#   20% para teste
######################################
SEED = 8188
x_train, x_test, y_train, y_test = train_test_split(tweets, classificacao, test_size=.2, random_state=SEED)

print("Total de instâncias de treino {0} com {1:.2f}% Negativo, {2:.2f}% Positivo, {2:.2f}% Neutro".format(
        len(x_train), 
        (len(x_train[y_train == 'Positivo']) / (len(x_train)*1.))*100,
        (len(x_train[y_train == 'Negativo']) / (len(x_train)*1.))*100,
        (len(x_train[y_train == 'Neutro']) / (len(x_train)*1.))*100))

######################################
# Abordagem Bag of Words
# Cria vetores com a frequencia de ocorrencia das palavras
# Ex.:
# ["olá", "sim", "posso", "não"]
# [1,8,2,6] -> frequencia com que as palavras ocorrem e sua relação com a classificação (neutro, positivo, negativo)
######################################

#result = process_model.process_data_bag_of_words(tweets, classificacao, x_test, y_test, stopwords = stops, n_grams = 1)

#visualization.plot_prediction_result(result, title='Acurácia x Classificador x N-Grams', filename='accuracia_classificador_ngrams.png')

process_model.process_data_bag_of_words(x_train, y_train, x_test, y_test, stopwords = stops, n_grams = 3)

##########################################
# Plota gráficos para análise dos dados
##########################################
if generate_visualization:
    visualization.plot_dataset_class_distribution(dataset)
    
    #visualization.wordcloud(dataset, stops)