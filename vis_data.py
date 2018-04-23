# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 17:30:00 2018

@author: evert

Exibe uma visualização para os dados
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


from wordcloud import WordCloud
"""
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 10
plt.rcParams["figure.figsize"] = fig_size
mpl.rcParams['font.size']=12                #10 
#mpl.rcParams['savefig.dpi']=50             #72 
mpl.rcParams['figure.subplot.bottom']=.1 
"""
def distr_qtd_carac(dataset):
    """
    Boxplot com a distribuição da quantidade de caracteres em cada tweet
    
    Keyword arguments:
        dataset -- dataset com os dados  
    """
    dataset['TamanhoTexto'] = [len(texto) for texto in dataset.Text]
    
    fig, ax = plt.subplots(1)
    plt.title('Qtd. caracteres x Tweets')
    ax.set_ylabel('Qtd. Caracteres')
    plt.boxplot(dataset.TamanhoTexto, labels=['Tweet'])
    #plt.show()
    plt.savefig('./data_visualization/dist_tamanho_texto.png')
    plt.close()


def wordcloud(dataset, stops):
    """
    Cria nuvem de palavras para as três classes: Positivo, Negativo e Neutro
    
    Keyword arguments:
        dataset -- dataset com os dados  
          
        stops -- Stopwords
    """
    ######################################
    # Cria nuvem de palavras para os registro do tipo Positivo
    ######################################
    
    wc_positivo = WordCloud(width=1600, height=800, max_font_size=200, colormap='summer', background_color='white', stopwords=stops, max_words=40).generate(' '.join(dataset[dataset.Classificacao == 'Positivo'].Text))
    wc_negativo = WordCloud(width=1600, height=800, max_font_size=200, colormap='autumn', background_color='white', stopwords=stops, max_words=40).generate(' '.join(dataset[dataset.Classificacao == 'Negativo'].Text))
    wc_neutro   = WordCloud(width=1600, height=800, max_font_size=200, colormap='PuBu', background_color='white', stopwords=stops, max_words=40).generate(' '.join(dataset[dataset.Classificacao == 'Neutro'].Text))
    
    # Generate plot
    fig = plt.figure(figsize=(12,10))
    plt.imshow(wc_negativo, interpolation="bilinear")
    plt.axis("off")
    #plt.show()
    fig.savefig("./data_visualization/word_cloud_neg.png")
    plt.close()
    
    fig = plt.figure(figsize=(12,10))
    plt.imshow(wc_positivo, interpolation="bilinear")
    plt.axis("off")
    #plt.show()
    fig.savefig("./data_visualization/word_cloud_pos.png")
    plt.close()
    
    fig = plt.figure(figsize=(12,10))
    plt.imshow(wc_neutro, interpolation="bilinear")
    plt.axis("off")
    #plt.show()
    fig.savefig("./data_visualization/word_cloud_neutro.png")
    plt.close()


def plot_dataset_class_distribution(dataset):
    dist = dataset.Classificacao.value_counts()
    
    y_pos = np.arange(len(dist))
    
    fig, ax = plt.subplots(figsize=(12,10))
    
    plt.title('Distribuição das classes')
    plt.ylabel('Frequência')
    
    plt.xlabel('Classes')

    colors = ['b', 'r', 'g']

    bar = plt.bar(y_pos, dist.values, align='center', alpha=0.5, color=colors)
    plt.xticks(y_pos, dist.index,rotation='vertical')

    i = 0
    for rect in bar:
        val = dist.iloc[i]
        height = rect.get_height()
        xloc = rect.get_x() + rect.get_width()/2.0
        yloc = 1.02*height
        #ax.text(xloc,1.02*height, val)
        ax.text(xloc, yloc, val, horizontalalignment='center',
                         weight='bold',
                         clip_on=True)
        i = i + 1
    
    plt.savefig("./data_visualization/class_distrib.png")
    plt.close()
    """
    
    plt.figure(figsize=(12,10))
    
    plt.bar(y_pos, term_freq_df2.sort_values(by='positive', ascending=False)['positive'][:50], align='center', alpha=0.5)
    plt.xticks(y_pos, term_freq_df2.sort_values(by='positive', ascending=False)['positive'][:50].index,rotation='vertical')
    
    plt.ylabel('Frequency')
    plt.xlabel('Top 50 positive tokens')
    plt.title('Top 50 tokens in positive tweets')
    """
def plot_prediction_result(result):
    
    y_pos = np.arange(len(result))
    
    fig, ax = plt.subplots(figsize=(12,10))
    
    plt.title('Acurácia x Classificador')
    plt.ylabel('Acurácia')
    
    plt.xlabel('Classificador')

    colors = ['blue', 'red', 'orange']

    bar = plt.bar(y_pos, result['Acurácia'], align='center', alpha=0.5, color=colors)
    plt.xticks(y_pos, result['Classificador'],rotation='vertical')

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
    
    plt.savefig("./data_visualization/accuracy_x_model.png")
    plt.show()
    #plt.close()
    