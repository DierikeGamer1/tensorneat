import os

def returnultimoCheckPoint(diretorio):
    # Lista todos os arquivos no diretório
    arquivos = os.listdir(diretorio)
    
    # Filtra apenas os arquivos, ignorando diretórios
    arquivos = [os.path.join(diretorio, arquivo) for arquivo in arquivos if os.path.isfile(os.path.join(diretorio, arquivo))]
    
    # Obtém a data da última modificação de cada arquivo
    infos_arquivos = [(arquivo, os.path.getmtime(arquivo)) for arquivo in arquivos]
    
    # Ordena os arquivos pela data da última modificação
    arquivos_ordenados = sorted(infos_arquivos, key=lambda x: x[1], reverse=True)
    
    # Retorna o arquivo mais recentemente modificado
    if arquivos_ordenados:
        return arquivos_ordenados[0][0]
    else:
        return None

# # Diretório que você quer verificar
# diretorio = './CheckPoints'

# # Obtém o arquivo mais recentemente modificado
# arquivo_recente = ultimoCheckPoint(diretorio)

# if arquivo_recente:
#     print("Arquivo mais recente:", arquivo_recente[0])
# else:
#     print("Nenhum arquivo encontrado no diretório.")
