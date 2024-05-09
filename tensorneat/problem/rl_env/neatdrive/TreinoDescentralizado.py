import neat
import json
import numpy as np
from datetime import datetime
from multiprocessing import pool
import multiprocessing
import os
import time
import classe
import sys
import requests
import socket
import random
import pickle
from neat.reporting import ReporterSet
from neat.math_util import mean
from neat.six_util import iteritems, itervalues
import subprocess
import threading


def kill_process_using_port(port):
    subprocess.run(["fuser", "-k", f"{port}/tcp"])


class CompleteExtinctionException(Exception):
    pass


HOST = "127.0.0.1"  # Endereço IP do servidor

ConfigEGenome_Port = 10330
ReturnGenome_Port = 65430


num_generations = 500
num_episodes = 1
cont = 0
ListaVchloE = []
ListaClosePrice = []
Listadados = []
ListaIp = ['165.232.152.41','64.23.185.216','64.227.100.178','64.23.133.89','64.23.141.224','64.23.141.26','64.23.141.149','146.190.151.111','64.23.141.154','64.23.189.220']
ListaIp = ['172.28.0.12']
ListaGenoma=[]
# Portas a serem liberadas
ports_to_free = [ConfigEGenome_Port, ReturnGenome_Port]


lista = os.listdir("DadosTreino")
for numero, nomearquivo in enumerate(lista):
    if "adausdt" in nomearquivo:
        with open(f"DadosTreino/{nomearquivo}", "r") as nomearquivo:
            Listadados.append(
                [
                    json.loads(linha.replace(
                        "'", '"').replace("+", ""))["data"]
                    for linha in nomearquivo
                ]
            )
            ListaClosePrice.append(
                [float(objeto["c"])
                 for objeto in Listadados[len(Listadados) - 1]]
            )


def VerificarFinals(Final, episode_reward):
    if Final:
        Final = True
    elif episode_reward < -10:
        Final = True
    return Final

def EnviarGenomeEConfigPeloIp(ip,genomes,config):
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                # Conecte-se ao servidor
                s.connect((ip, ConfigEGenome_Port))
                data=[genomes,config]
                data=pickle.dumps(data)
                s.sendall(data)
                print(f"Genomas e Config enviados com sucesso.{ip}")
                break
        except:
            pass
    
   




def EnviarListaDeGnomasEConfigParaOsPools(ListaIp,genomes,config):
    threads = []
    ListaGenomas=DividirGenomas(genomes,ListaIp)
    cont=0
    for ip in ListaIp:
        thread = threading.Thread(target=EnviarGenomeEConfigPeloIp, args=(ip,ListaGenomas[cont],config))
        thread.start()
        threads.append(thread)
        cont+=1

    # Aguardar todas as threads terminarem
    for thread in threads:
        thread.join()
    print(f"Todos Genomas e Config enviados com sucesso.")



def EsperaGenomaPeloIp(ip):
    global ListaGenoma
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                # Conecte-se ao servidor
                s.connect((ip, ReturnGenome_Port))
                print(f"Conectado Para Receber Genomas De Volta Ip:{ip}")
                print(f"Aguardando Genomas... Ip{ip}")
                received_data = b""
                while True:
                    # Receba os dados do servidor em blocos
                    data = s.recv(1024)
                    if not data:
                        break
                    received_data += data
                print("Genomas Recebidos!")
                for g in pickle.loads(received_data):
                    ListaGenoma.append(g)
                s.close()
                break
        except:
            pass


def EsperarGenomas(ListaIp):
    global ListaGenoma

    # Iniciar uma thread para cada IP na lista
    threads = []
    ListaGenoma=[]
    for ip in ListaIp:
        thread = threading.Thread(target=EsperaGenomaPeloIp, args=(ip,))
        thread.start()
        threads.append(thread)

    # Aguardar todas as threads terminarem
    for thread in threads:
        thread.join()
    
    return ListaGenoma
def DividirGenomas(ListaGenomas,ListaIp):
    # Calcula o tamanho de cada parte
    tamanho_parte = len(ListaGenomas) // len(ListaIp)
    sobra = len(ListaGenomas) % len(ListaIp)  # Calcula a sobra

    # Lista para armazenar as partes para cada pessoa
    partes_por_pessoa = []

    inicio = 0
    for i in range(len(ListaIp)):
        # Calcula o fim do intervalo para esta pessoa
        fim = inicio + tamanho_parte + (1 if i < sobra else 0)
        
        # Adiciona a parte desta pessoa à lista de partes
        partes_por_pessoa.append(ListaGenomas[inicio:fim])
        
        # Atualiza o início para a próxima pessoa
        inicio = fim

    # Exibe as partes para cada pessoa
    Genomas=[]
    for i, partes in enumerate(partes_por_pessoa):
        # print(f'Ip {ListaIp[i-1]}: {partes}')
        Genomas.append(partes)
    return Genomas

def eval_genome(genomes, config):
    global ListaIp
    EnviarListaDeGnomasEConfigParaOsPools(ListaIp,genomes, config)
    genomas = EsperarGenomas(ListaIp)  
    try:
        for id, g in genomes:  
            for id2, g2 in genomas:
                if id2 == id:
                    g.fitness = g2.fitness
                    break
    except:
        print(genomas)
        print(id)
        print("Erro")

        sys.exit()
    print(len(genomes))
    return genomes


def run(config_file, CarregarWin=True, CarregarCheckPoint=False):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )

    if CarregarCheckPoint:
        p = neat.Checkpointer.restore_checkpoint(
            "CheckPoints/neat_2024-02-2022-29-45checkpoint1440"
        )

    else:
        p = neat.Population(config)
    # Adicionar um reporter de saída padrão para mostrar o progresso no terminal
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    checkpoint_path = f"CheckPoints/neat_{str(datetime.now())[:-7].replace(' ', '').replace(':','-')}checkpoint"
    p.add_reporter(
        neat.Checkpointer(generation_interval=1,
                          filename_prefix=checkpoint_path)
    )
    if CarregarWin:
        winner_file = "genomavencedor/bestGenome0.147499999999999662024-02-2122-37-10.pkl"
        with open(winner_file, "rb") as f:
            winner = pickle.load(f)
        # Ensure there is at least one species.
        if not p.species.species:
            # Create a new species if the list is empty
            p.species.create_new_species()
        # Replace the champion genome in the new population with the saved winner genome.
        if p.species.species:
            winner_dict = {winner.key: winner}
            p.species.species[1].members = winner_dict

    winner = p.run(eval_genome, num_generations)

    # Salvar o genoma vencedor
    with open(
        f"genomavencedor/genoma_vencedor{str(datetime.now())[:-7].replace(' ', '').replace(':','-')}.pkl",
        "wb",
    ) as arquivo:
        pickle.dump(winner, arquivo)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "configteste.txt")

    # Iniciar o processo

    run(config_path)
