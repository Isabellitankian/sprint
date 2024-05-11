import csv
import requests
from fastapi import FastAPI

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

app = FastAPI()

@app.get('/')
def get_todos_arquivos():
    return request_embrapa()

@app.get('/{arquivo:str}')
def get_arquivo(arquivo:str):
    return request_embrapa(arquivo)