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
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from nltk.corpus import stopwords

from wordcloud import WordCloud

from bs4 import BeautifulSoup

from pprint import pprint

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 10
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
mpl.rcParams['font.size']=12                #10 
#mpl.rcParams['savefig.dpi']=50             #72 
mpl.rcParams['figure.subplot.bottom']=.1 


#carrega o dataset de tweets
df = pd.read_csv('./input/Tweets_Mg.csv', encoding='utf-8')

df['TamanhoTexto'] = [len(text) for text in df.Text]

###################################################
# Preparando dicionário de dados para análise
###################################################
data_dict = {
    'sentiment':{
        'description':'Classe de sentimento - Negativo, Positivo, Neutro'
    },
    'text':{
        'description':'Texto do tweet'
    },
    'pre_clean_len':{
        'description':'Tamanho do texto antes da limpeza'
    },
    'dataset_shape':df.shape
}

fig, ax = plt.subplots(figsize=(5, 5))
plt.title('Qtd. caracteres x Tweets')
ax.set_ylabel('Qtd. Caracteres')
plt.boxplot(df.TamanhoTexto, labels=['Tweet'])
plt.show()
plt.savefig('./data_visualization/dist_tamanho_texto.png')
plt.close()
