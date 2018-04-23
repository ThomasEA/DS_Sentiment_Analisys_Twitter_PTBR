# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 16:54:48 2018

@author: evert

 Rotinas para limpeza dos dados
"""
import nltk
import numpy as np
from bs4 import BeautifulSoup

##Faz o download das stopwords
nltk.download('stopwords')
nltk.download('rslp')

def clean_data(dataset, shuffle = False):
    """
    Limpa a coluna de texto do Tweet, retirando caracteres HTML, links, menções, etc..
    
    Keyword arguments:
        dataset -- o dataset que possui a coluna de texto
    
        shuffle -- faz um shuffle dos dados (default = False)
    """
    df = dataset.copy()
    
    #1. Retira HTTP links
    df.Text.replace(r"http\S+", "", regex=True, inplace=True)
    
    #2. Retira tags e caracteres HTML
    df.Text = [BeautifulSoup(text, 'lxml').get_text() for text in df.Text]
    
    #3. Retira menções (@xxxxxx)
    df.Text.replace(r"@[A-Za-z0-9]+", "", regex=True, inplace=True)
    
    #4. !!! Importante !!!
    #Optei por deixar hashtags, por imaginar que possam trazer alguma orientação
    #relacionada ao tweet
    
    #5. Transforma o texto em lowecase
    #http://each.uspnet.usp.br/digiampietri/BraSNAM/2017/p04.pdf
    #RP de Pelle Pelle, VPM Moreira
    #Segundo este artigo pode trazer benefícios
    df.Text = df.Text.str.lower()
    
    df.Text.replace('', np.nan, inplace=True)
    df.dropna(subset=['Text'], inplace=True)
    
    ######################################################################################
    # Faz um shuffle dos dados
    #
    #   IMPORTANTE: SÓ COM O SHUFFLE GANHA 5% DE ACURÁCIA E RECALL/PRECISION VÃO A 0.96
    ######################################################################################
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    
    return df

def apply_text_processing(dataset):
    """
    Aplica as rotinas de processamento de texto, como stemming e montagem do dicionário
    de stopwords
    
    Keyword arguments:
        dataset -- o dataset que possui a coluna de texto
        
    Return:
        dataset -- o dataset com a coluna de texto tratada
        
        stopwords -- array de stopwords
    """
    stops = nltk.corpus.stopwords.words('portuguese')
    
    stemmer = nltk.stem.RSLPStemmer()
    dataset.Text = [stemmer.stem(t) for t in dataset.Text]
    
    return dataset, stops