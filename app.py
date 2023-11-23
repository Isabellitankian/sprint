import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics


import streamlit as st
import matplotlib.pyplot as plt


class Crawler:


    def _init_(self) -> None:
        self.breast = load_breast_cancer()

        X = self.breast.data
        y = self.breast.target

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X,
                                                     y,
                                                     test_size=0.3,
                                                     random_state=42)
        
        self.modelo = LogisticRegression(max_iter=4000)
        self.modelo.fit(self.X_train, self.Y_train)


    def get_def(self):
        
        breast_data = self.breast.data
        breast_labels = self.breast.target

        labels = np.reshape(breast_labels, (569, 1))
        final = np.concatenate( [breast_data, labels], axis=1 )
        
        colunas = list(self.breast.feature_names)
        colunas.append("label")

        df = pd.DataFrame(final, columns=colunas)
        
        return df

    
    def relatorio_classification(self):

        y_predict = self.modelo.predict(self.X_test)
        return metrics.classification_report(self.Y_test, y_predict)
    
    
    def modelo_regressao(self):

        st.write()

        y_predict = self.modelo.predict(self.X_test)

        fig, ax = plt.subplots()
        metrics.ConfusionMatrixDisplay.from_predictions(self.Y_test, y_predict)
        # sns.heatmap(matriz_confusao, annot=True)
        st.write(fig)


    def get_acuracia(self):

        y_predict = self.modelo.predict(self.X_test)
        return metrics.accuracy_score(self.Y_test,y_predict )

    # def plot_confusion_matrix(X_train, y_train):
    #     labels=list(map(np.argmax,y_train))
    #     labels_pred = list(map(np.argmax, y_train))

    #     cf_matrix = confusion_matrix(labels, labels_pred)
    #     sns.heatmap(cf_matrix, annot=True)



op = Crawler()


def main():


    st.set_page_config(page_title = 'S3 - Automated Machine Learning and Deploy',

                        layout = 'wide',
                        initial_sidebar_state = 'expanded')

    

    st.title("""
        Analise de imagens de pessoas com câncer, divididas entre beligno e maligno. 

        Por que utilizar Logistic Regreesion?

        A regressão logística é uma técnica estatística frequentemente utilizada para realizar classificação binária, onde o objetivo é prever se uma observação pertence a uma das duas categorias possíveis. Quando aplicada ao campo médico, especificamente na classificação de imagens de câncer como benigno ou maligno, a regressão logística apresenta várias vantagens e importâncias.
             
        1. Interpretabilidade:
            A regressão logística produz coeficientes que podem ser interpretados como log-odds. Esses coeficientes indicam a direção e magnitude da influência de cada característica na probabilidade de um caso ser classificado como maligno. Isso é crucial no contexto médico, pois os médicos podem entender quais características da imagem estão associadas a um maior risco de câncer.
        
        2. Probabilidades ajustadas:
            A regressão logística fornece probabilidades ajustadas, que são úteis para avaliar o quão certa ou incerta é uma predição. No caso de imagens médicas, é valioso ter uma medida de confiança na classificação, pois isso pode impactar decisões subsequentes sobre o tratamento ou acompanhamento.
    
        3. Lidar com dados desbalanceados:
            Em problemas médicos, os conjuntos de dados muitas vezes são desbalanceados, com uma classe sendo mais prevalente do que a outra. A regressão logística é capaz de lidar com esse desbalanceamento, ajustando os pesos durante o treinamento para garantir uma consideração apropriada de ambas as classes.

        4. Eficiência computacional:
            Comparada a modelos mais complexos, como redes neurais profundas, a regressão logística é computacionalmente eficiente. Isso é particularmente importante em aplicações médicas, onde a interpretabilidade e eficiência de recursos são cruciais.

        5. Facilidade de implementação:
            A regressão logística é relativamente fácil de entender e implementar. Isso facilita a integração da técnica em ambientes clínicos, onde a compreensão do modelo é crucial para a aceitação e confiança dos profissionais de saúde.      

        6. Regularização:
            A regressão logística pode ser regularizada para evitar overfitting. Em situações médicas, onde os conjuntos de dados podem ser pequenos, a capacidade de evitar o overfitting é vital para garantir que o modelo generalize bem para novos dados.


    """)
    

    with st.expander('', expanded = True):
        
        st.title('')
        st.title('Dados de indivíduos com câncer')

        df_plot = op.get_def()
        st.dataframe(df_plot)
    
    st.title("""
        Matriz de Confusão:
             
        A matriz de confusão é uma ferramenta crucial na avaliação de modelos, incluindo a regressão logística. Ela fornece uma visão resumida e intuitiva do desempenho do modelo ao comparar suas previsões com os resultados reais. Os elementos da matriz, como verdadeiros positivos, verdadeiros negativos, falsos positivos e falsos negativos, são essenciais para calcular métricas como precisão, recall, especificidade e a pontuação F1.
        
        Essa análise mais detalhada ajuda a entender não apenas a taxa de acertos gerais, mas também como o modelo lida com diferentes tipos de erros, fornecendo informações valiosas para ajustes e melhorias.
             
        Abaixo a matriz do resultado que o modelo previu.
    """)
    
    with st.expander('', expanded = True):
        
        st.title('Matriz de comparação de métricas')

        acuracia = op.relatorio_classification()
        st.text(acuracia)
    
    
    with st.expander('', expanded = True):
        
        st.title('Acuracia do Modelo')

        
        acuracia = op.get_acuracia()
        st.title(f'Acurácia: {round(acuracia * 100, 2)}%')
        

    

            
           


        


if _name_ == '_main_':
  
    main()