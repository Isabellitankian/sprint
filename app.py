import streamlit as st
from pandas import read_csv
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import time
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk




class Crawler:


    def __init__(self) -> None:
        pass


    def execute(self):

        html = self.get_html_site('https://www.imdb.com/chart/moviemeter/?ref_=chttvtp_ql_2')

        links_sinopses = self.get_links_sinopses(html)
        print(f'N° de listas: {len(links_sinopses)}')

        df_sinopse = self.get_df_sinopse(links_sinopses)

        return df_sinopse



    def get_html_site(self, url):
        userAgents=[
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/74.0.3729.157 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/12.1.1 Safari/605.1.15"]

        url = 'https://www.imdb.com/chart/moviemeter/?ref_=chttvtp_ql_2'
        response = requests.get(url, headers={"User-agent": userAgents[1]})
        
        return BeautifulSoup(response.text, "html.parser")
    

    def get_df_sinopse(self, links_sinopse):
        headers = {
            'authority': 'www.amazon.com.br',
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'accept-language': 'pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7',
            'cache-control': 'max-age=0',
            'device-memory': '8',
            'downlink': '10',
            'dpr': '1.875',
            'ect': '4g',
            'rtt': '50',
            'sec-ch-device-memory': '8',
            'sec-ch-dpr': '1.875',
            'sec-ch-ua': '"Chromium";v="110", "Not A(Brand";v="24", "Google Chrome";v="110"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-ch-ua-platform-version': '"10.0.0"',
            'sec-ch-viewport-width': '455',
            'sec-fetch-dest': 'document',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-site': 'none',
            'sec-fetch-user': '?1',
            'upgrade-insecure-requests': '1',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36',
            'viewport-width': '455',
        }
        list_genre = []
        list_title_pt = []
        list_year = []
        list_sinopse = []

        for link in links_sinopse:
            time.sleep(.5)
            response = requests.get(link, headers = headers)
            html = response.content
            soup = BeautifulSoup(html, "html.parser")

            #genre
            try:
                for genre in soup.find('span', {'class':'ipc-chip__text'}):
                    genre = genre.text
                    list_genre.append(genre)
            except:
                list_genre.append(np.nan)
                print('Nan')

            #title_PT and year
            try:
                for x in soup.find('title'):
                    #title_pt
                    title_pt = (x.text)[:-14].strip()
                    list_title_pt.append(title_pt)
                    #year
                    year = (x.text)[-12:-8].strip()
                    list_year.append(year)

            except:
                list_title_pt.append(np.nan)
                list_year.append(np.nan)

            #sinopse
            try:
                for sin in soup.find('span', {"data-testid":"plot-xl"}):
                    sinopse = sin.text
                    list_sinopse.append(sinopse)
            except:
                list_sinopse.append(np.nan)

        a = {'title_pt' : list_title_pt ,'year' : list_year , 'genre': list_genre , 'sinopse':list_sinopse}
        df = pd.DataFrame.from_dict(a, orient='index')
        return df.transpose()
        

    def get_links_sinopses(self, html):
        list_links = []
        for a in html.find_all('a', href=True):
            if '/title/' in a['href'] and 'https://www.imdb.com/'+ a['href'] not in list_links:
                list_links.append(('https://www.imdb.com/'+a['href'])[:-15])

        #Remove duplicates
        list_links = list(dict.fromkeys(list_links))
        #Deleting first element
        list_links = list_links[1:]

        return list_links


    def salva_csv(self):
        df = self.execute()
        df.to_csv('sinopse.csv', index=False, encoding='utf-8-sig')


def get_kmeans_labels(df_processed):

    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.95)
    X = vectorizer.fit_transform(df_processed['sinopse_no_stopwords'])

    # initialize kmeans with 5 centroids
    kmeans = KMeans(n_clusters=5, random_state=42)
    # fit the model
    kmeans = kmeans.fit(X)
    #predicting the clusters and store cluster labels in a variable
    return kmeans.predict(X)

    

def processa_dados(dados):

    def qty_words(text):
        words= text.split()
        word_count = len(words)
        return word_count

    df_processed = dados.copy()
    df_processed['sinopse'] = df_processed['sinopse'].str.lower()

    ### Feature Engineering
    df_processed['word_count'] = df_processed['sinopse'].apply(qty_words).astype('int64')

    nltk.download('stopwords')
    stopwords = nltk.corpus.stopwords.words('portuguese')
    df_processed['sinopse_no_stopwords'] = df_processed['sinopse'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))

    df_processed['cluster'] = get_kmeans_labels(df_processed)

    return df_processed


def get_df_plot(df_processed):
  
    df_plot = df_processed.groupby(['cluster', 'genre'])['title_pt'].count()
    df_plot = df_plot.reset_index()
    df_plot.rename(columns = {'title_pt':'count'}, inplace = True)
    df_plot['%'] = 100 * df_plot['count'] / df_plot.groupby('cluster')['count'].transform('sum')
    df_plot = df_plot.sort_values(['cluster', '%'], ascending = False).groupby('cluster').head(11)

    return df_plot


def word_cloud_cluster(df_processed, cluster: int):
  
  text = ' '.join([phrase for phrase in df_processed.loc[df_processed.cluster == cluster]['sinopse_no_stopwords']])
  plt.figure(figsize=(7,5), facecolor='None')
  
  wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)

  plt.imshow(wordcloud, interpolation='bilinear')
  plt.axis('off')
  plt.title(f'Cluster "{cluster}" | Palavras mais frequentes', fontsize = 19)
  
  st.pyplot(plt.gcf())
  



def main():


    st.set_page_config(page_title = 'S3 - Automated Machine Learning and Deploy',

                        layout = 'wide',
                        initial_sidebar_state = 'expanded')

    st.title('Exibição de filmes por cluster :D')

    with st.sidebar:
        c1, c2 = st.columns(2)

        c2.write('')
        c2.subheader('Automated Machine Learning and Deploy')

        # database = st.selectbox('Fonte dos dados de entrada (X):', ('CSV', 'Online'))
        database = st.radio('Fonte dos dados de entrada (X):', ('CSV', )) #, 'Online'
        df_sinopse = None

        if database == 'CSV':
            st.info('Upload do CSV')
            file = st.file_uploader('Selecione o arquivo CSV', type='csv')
            if file:
                df_sinopse = read_csv(file)
            
                

        # elif database == 'Online':
        #     crawler = Crawler()
        #     df_sinopse = crawler.execute()
    

    
    if df_sinopse is not None:

        # st.table(df_sinopse)
        with st.expander('Filmes por genêro:', expanded = True):

            fig = px.bar(df_sinopse.genre.value_counts('d')*100,
                text_auto=True,
                title = '% de Filmes por Gênero',
                labels={'index':'Gênero',
                        'value':'% de Filmes'})
            
            st.plotly_chart(fig)
        
        
        with st.expander('Filmes por ano:', expanded = True):

            plt.figure(figsize = (20,7))
            sns.histplot(df_sinopse,
                        x = 'year',
                        kde = True).set_title('Qtd de Filmes por Ano')
            plt.xticks(rotation=45)
            st.pyplot(plt.gcf())
        
        df_processado = processa_dados(df_sinopse)
        print(df_processado)

        with st.expander('Generos por clusters:', expanded = True):
            
            df_plot = get_df_plot(df_processado)

            plt.figure(figsize = (20, 10))
            g = sns.catplot(
                data=df_plot,
                x="genre",
                y = "%",
                col="cluster",
                kind="bar",
                height=4,
                aspect=1,
                sharex = False
            )
            
            # fig.set_xlabels('')
            g.set_xticklabels(rotation=90, size = 8)

            st.pyplot(plt.gcf())
        
        
        
        with st.expander('Visualizar gráfico solar teste:', expanded = True):
            
            
            for c in range(5):
                word_cloud_cluster(df_processado, c)
            
           


        


if __name__ == '__main__':
  
    main()