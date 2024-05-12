import csv
import requests
import streamlit as st
import matplotlib.pyplot as plt


class Crawler:

    def request_embrapa(arquivo = None):
        urlBase = 'http://vitibrasil.cnpuv.embrapa.br/download/'
        arquivos = ['Producao', 'ProcessaViniferas', 'ProcessaAmericanas', 'ProcessaMesa', 'ProcessaSemclass', 'Comercio', 'ImpVinhos', 
                    'ImpEspumantes', 'ImpFrescas', 'ImpPassas', 'ImpSuco', 'ExpVinho', 'ExpEspumantes', 'ExpUva', 'ExpSuco']   
        resposta = {}
        if arquivo == None:
            for arquivo in arquivos:
                url = f'{urlBase}{arquivo}.csv'
                response = requests.get(url)
                if response.status_code != 200:
                    resposta = {'erro': response.status_code}
                    break
                reader = csv.DictReader(response.text.splitlines())
                resposta[arquivo] = []
                for linha in reader:
                    resposta[arquivo].append(linha)
            return resposta
        elif arquivo in arquivos:
                url = f'{urlBase}{arquivo}.csv'
                response = requests.get(url)
                if response.status_code != 200:
                    resposta = {'erro': response.status_code}
                else:
                    reader = csv.DictReader(response.text.splitlines())
                    resposta[arquivo] = []
                    for linha in reader:
                        resposta[arquivo].append(linha)
        else:
            resposta = {'erro': 404} 
        return resposta



op = Crawler()


def main():


    # st.set_page_config(page_title = 'Request API - The Outliers',

    #                     layout = 'wide',
    #                     initial_sidebar_state = 'expanded')

    

    # st.title("""
        
    #         Olá! O intutito desse request é fazer o download no site da embrapa e realizar o download dos CSVs que são de importância para nós.


    # """)
    
    st.set_page_config(
    page_title="Olá! O intutito desse request é fazer o download no site da embrapa e realizar o download dos CSVs que são de importância para nós.",
    page_icon="👋",
    )

    st.write("# Olá! 👋👋👋 Bem vindo ao demonstrativo do site da embrapa, no qual fazemos o request de suas APIs!")

    st.sidebar.success("Estamos em obras aqui, peço que espere um pouquinho.")

    st.markdown(
        """
        - Site da Emprapa:  http://vitibrasil.cnpuv.embrapa.br

        - Contém os dados da produção:  http://vitibrasil.cnpuv.embrapa.br/index.php?opcao=opt_02

        - Contém os dados do processamento:  http://vitibrasil.cnpuv.embrapa.br/index.php?opcao=opt_03

        - Contém os dados da comercialização:  http://vitibrasil.cnpuv.embrapa.br/index.php?opcao=opt_04

        - Contém os dados da importação: http://vitibrasil.cnpuv.embrapa.br/index.php?opcao=opt_05

        - Contém os dados da Exportação:  http://vitibrasil.cnpuv.embrapa.br/index.php?opcao=opt_06
        
    """
    )
    # with st.expander('', expanded = True):
        
    #     st.title('')
    #     st.title('Dados de indivíduos com câncer')

    #     df_plot = op.get_def()
    #     st.dataframe(df_plot)
    #
    
    # st.title("""
    #     Matriz de Confusão:
             
    #     A matriz de confusão é uma ferramenta crucial na avaliação de modelos, incluindo a regressão logística. Ela fornece uma visão resumida e intuitiva do desempenho do modelo ao comparar suas previsões com os resultados reais. Os elementos da matriz, como verdadeiros positivos, verdadeiros negativos, falsos positivos e falsos negativos, são essenciais para calcular métricas como precisão, recall, especificidade e a pontuação F1.
        
    #     Essa análise mais detalhada ajuda a entender não apenas a taxa de acertos gerais, mas também como o modelo lida com diferentes tipos de erros, fornecendo informações valiosas para ajustes e melhorias.
             
    #     Abaixo a matriz do resultado que o modelo previu.
    # """)
    
    # with st.expander('', expanded = True):
        
    #     st.title('Matriz de comparação de métricas')

    #     acuracia = op.relatorio_classification()
    #     st.text(acuracia)
    
    
    # with st.expander('', expanded = True):
        
    #     st.title('Acuracia do Modelo')

        
    #     acuracia = op.get_acuracia()
    #     st.title(f'Acurácia: {round(acuracia * 100, 2)}%')
        

    



if __name__ == '__main__':
  
    main()