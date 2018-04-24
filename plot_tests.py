# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 01:46:31 2018

@author: evert

Testes de plotagens
"""
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_prediction_result_by_ngrams(result, title='', filename='accuracy_x_model.png'):
    
    y_pos = np.arange(len(result))
    width = 0.20       # the width of the bars
    
    fig = plt.figure(figsize=(12,10))
    
    ax = fig.add_subplot(1,1,1)
    
    major_ticks = np.arange(0, 101, 5)                                              
    minor_ticks = np.arange(0, 101, 2.5)
    
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    
    ax.set_xticks(y_pos + width / 2)
    ax.set_xticklabels(result['NGrams'])
    
    plt.title(title)
    plt.ylabel('Acurácia')
    
    plt.xlabel('Classificador')

    colors = ['blue', 'red', 'orange']

    x = result.loc[result['NGrams'] == 'unigram'][['NGrams', 'Acurácia']]
    
    print(x)
    
    bar = ax.bar(y_pos, x['Acurácia'], align='center', alpha=0.5, color=colors, yerr=result['F1Score'])
    plt.xticks(y_pos, x['NGrams'],rotation='vertical')

    i = 0
    for rect in bar:
        val = result.iloc[i]['Acurácia']
        height = rect.get_height()
        xloc = rect.get_x() + rect.get_width()/2.0
        yloc = 1.02*height
        #ax.text(xloc,1.02*height, val)
        ax.text(xloc, yloc, '%.2f%%' % val, horizontalalignment='center',
                         weight='bold',
                         clip_on=True)
        i = i + 1
    
    axes = plt.gca()
    axes.set_ylim([70,100])
    
    #plt.savefig("./data_visualization/" + filename)
    plt.show()

d = {'Classificador': ['Naive Bayes', 'SVM', 'Naive Bayes', 'SVM'],
     'Acurácia': [89, 90, 93, 96],
     'F1Score': [0.93, 0.95, 0.94, 0.96],
     'NGrams': ['unigram', 'unigram', 'bigram', 'bigram']}
     
dataset = pd.DataFrame(d)

plot_prediction_result_by_ngrams(dataset)