# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 17:57:04 2018

@author: evert

Rotinas para modificação, consulta e filtragem de dataset
"""

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

#Imprime a quantidade de registros por classe
#print(dataset.Classificacao.value_counts())

