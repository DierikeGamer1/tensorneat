import neat,json
import time
import os,sys
import classe
import socket
import pickle
import ast
from multiprocessing import Pool
import multiprocessing
from neat.six_util import iteritems, itervalues
import numpy as np
import subprocess


def kill_process_using_port(port):
    executar_comando(f'kill -9 {port}')

def executar_comando(cmd):
    resultado = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if resultado.returncode == 0:
        return resultado.stdout
    else:
        return resultado.stderr

# Defina o endereço IP e a porta em que o servidor irá ouvir
HOST = '0.0.0.0'  # Endereço IP local

ConfigEGenome_Port = 10330
ReturnGenome_Port = 65430
ports_to_free = [ ConfigEGenome_Port, ReturnGenome_Port]

ListaVchloE = []
ListaClosePrice = []
Listadados = []
for port in ports_to_free:
    kill_process_using_port(port)
    

lista = os.listdir("DadosTreino")
for numero, nomearquivo in enumerate(lista):
    if "adausdt" in nomearquivo:
        with open(f"DadosTreino/{nomearquivo}", "r") as nomearquivo:
            Listadados.append([json.loads(linha.replace("'", '"').replace(
                "+", ""))["data"] for linha in nomearquivo])
            ListaClosePrice.append([float(objeto["c"])
                                   for objeto in Listadados[len(Listadados)-1]])
            


class ParallelEvaluator(object):
    def __init__(self, num_workers, eval_function, timeout=None):
        """
        eval_function should take one argument, a tuple of
        (genome object, config object), and return
        a single float (the genome's fitness).
        """
        self.num_workers = num_workers
        self.eval_function = eval_function
        self.timeout = timeout
        self.pool = Pool(num_workers)

    def __del__(self):
        self.pool.close() # should this be terminate?
        self.pool.join()

    def evaluate(self, genomes, config):
        jobs = []
        # print(genomes)
        for ignored_genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome, config)))

        # assign the fitness back to each genome
        for job, (ignored_genome_id, genome) in zip(jobs, genomes):
            genome.fitness = job.get(timeout=self.timeout)
        return genomes


def ReceberGenomasEConfig():

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Faça o bind do socket ao endereço e porta especificados
    s.bind((HOST, ConfigEGenome_Port))
    # Espere por conexões entrantes
    s.listen()
    conn, addr = s.accept()
    with conn:
        print('Genomas E Config Recebido')
        # Inicialize uma variável para armazenar os dados recebidos
        received_data=b''
        while True:
            # Receba os dados do cliente em blocos
            data = conn.recv(1024)
            if not data:
                break
            received_data += data 
        received_data = pickle.loads(received_data)
        genomes=received_data[0]
        config=received_data[1]

        return genomes,config


def AvaliarVariosDeVez(genomes,config):
    pe=ParallelEvaluator(multiprocessing.cpu_count(), EvalSingleMulti)
    genomes=pe.evaluate(genomes, config)
    return genomes

def AvaliarUmDescadaVez(genomes,config): 
    for genome_id, genome in genomes:
        contLong = 0
        contShort = 0
        total_reward = 0
        fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        episode_reward = 0
        for numero, arquivo in enumerate(ListaClosePrice):
            ClosePrice = ListaClosePrice[numero]
            dados = Listadados[numero]
            env = classe.AmbientDeTreino(ClosePrice)
            state = env.reset()
            for indice, objeto in enumerate(dados):
                if indice > 202:
                    action = np.argmax(net.activate(state))
                    reward, Final, state = env.actions(
                        action, ClosePrice, indice)
                    if action == 0:
                        contLong += 1
                    elif action == 2:
                        contShort += 1
                    state.extend(ClosePrice[indice - 200: indice])
                    episode_reward += reward
                    if VerificarFinals(Final, episode_reward):
                        break
                    if indice > 2000:
                        break
            total_reward += episode_reward
        fitness += total_reward
        if fitness > 0.01:
            if contLong>0 and contShort>0:
                fitness+=0.09
        genome.fitness = fitness
    return genomes


def VerificarFinals(Final, episode_reward):
    if Final:
        Final = True
    elif episode_reward < -10:
        Final = True
    return Final

def EvalSingleMulti(genome, config):
    contLong=0
    contShort=0
    total_reward = 0
    fitness = 0
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    episode_reward = 0
    for numero, arquivo in enumerate(ListaClosePrice):
        ClosePrice = ListaClosePrice[numero]
        dados = Listadados[numero]
        env = classe.AmbientDeTreino(ClosePrice)
        state = env.reset()
        for indice, objeto in enumerate(dados):
            if indice > 202:
                action = np.argmax(net.activate(state))
                reward, Final, state = env.actions(
                    action, ClosePrice, indice)
                if action == 0:
                    contLong += 1
                elif action == 2:
                    contShort += 1
                state.extend(ClosePrice[indice - 200: indice])
                episode_reward += reward
                if VerificarFinals(Final, episode_reward):
                    break
        total_reward += episode_reward
    fitness += total_reward
    if fitness>0.01:
        if contLong>0 and contShort>0:
            fitness+=0.09
    return fitness


def SocketParaRetorno(genomes):
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Faça o bind do socket ao endereço e porta especificados
            s.bind((HOST, ReturnGenome_Port))
            # Espere por conexões entrantes
            s.listen()
            conn, addr = s.accept()
            conn.sendall(pickle.dumps(genomes))
            s.close()
            print('Genomas enviados com sucesso.')
            break
        except:
            pass

    

if __name__ == "__main__":
    while True:
        
        genomes,config=ReceberGenomasEConfig()     
        print("Treinamento Iniciado!")
        genomes=AvaliarVariosDeVez(genomes,config)
        print("Acabou De Treinar!\nPreparando Para Enviar!")
        SocketParaRetorno(genomes) 
        print("Enviou!")
