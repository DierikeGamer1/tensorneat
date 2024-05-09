import neat
import pickle
import json
import time
from datetime import datetime
import multiprocessing
import os
import classe
import sys
import cython
from  ReturnUltimoCheckPoint  import *


num_generations = 10000
num_episodes = 1
cont = 0
ListaVchloE = []
ListaClosePrice = []
Listadados = []


def VerificarFinals(Final, indice, ListaClosePrice, env, SellPrice, episode_reward):
    reward = 0
    if Final:
        reward = env.FecharAcabouDados(SellPrice, indice)
    elif indice == ListaClosePrice - 1:
        Final = True
        reward = env.FecharAcabouDados(SellPrice, indice)
    elif episode_reward < -3:
        Final = True
        reward = env.FecharAcabouDados(SellPrice, indice)
    return Final, reward


def argmax_custom(lst):
    max_index, _ = max(enumerate(lst), key=lambda x: x[1])
    return max_index


def eval_genome(genome, config, ListaClosePrice):
    global num_episodes
    contLong: cython.int = 0
    contShort: cython.int = 0
    total_reward: cython.double = 0
    reward1: cython.double = 0
    fitness: cython.double = 0
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    episode_reward: cython.double = 0
    ultimoindice = 0
    ultimoindicelist=[]

    # with open("Processados.txt", "r") as arquivo:
    #     ListaClosePrice = [
    #         [
    #             float(v.replace("[", "").replace("]", "").replace(",", ""))
    #             for v in arquivo.read().split()
    #         ]
    #     ]

    for arquivo in ListaClosePrice:
        ClosePrice = arquivo
        dados = arquivo
        env = classe.AmbientDeTreino(ClosePrice)
        state = env.reset()
        for indice in range(len(dados)):
            if indice > 202:
                action = argmax_custom(net.activate(state))
                reward, Final, state = env.actions(action, ClosePrice)
                if action == 0:
                    contLong += 1
                elif action == 2:
                    contShort += 1
                state.extend(ClosePrice[indice - 200 : indice])
                episode_reward += reward

                Final, reward1 = VerificarFinals(
                    Final,
                    indice,
                    len(ListaClosePrice),
                    env,
                    ClosePrice,
                    episode_reward,
                )
                ultimoindice = indice
                if Final:
                    episode_reward += reward1
                    break
        ultimoindicelist.append(ultimoindice)
        fitness += episode_reward
    if contLong > len(ultimoindicelist)/3 and contShort > len(ultimoindicelist) and fitness>11:
        print(f"ContiLong:{contLong}")
        print(f"ContiShort:{contShort}")
        if contLong > len(ultimoindicelist)*5:
            contLong = len(ultimoindicelist)*5
        if contShort > len(ultimoindicelist)*5:
            contShort = len(ultimoindicelist)*5
        print(f"Reward Final:{reward1}")
        print(f"Antes Fitness:{fitness}  Indices:{ultimoindicelist}")
        print(f"Quantia De Inicios:{len(ultimoindicelist)}")
        fitness += 0.015 * contLong
        fitness += 0.015 * contShort
        for i in ultimoindicelist:
            fitness += i / 1000000
        fitness += 1
        print(f"Fitness:{fitness}  Indice:{ultimoindice}")
        print(f"Key:",genome.key)
        print("-" * 40)

    return fitness


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
            returnultimoCheckPoint('./CheckPoints')
        )
    else:
        p = neat.Population(config)
    # Adicionar um reporter de saída padrão para mostrar o progresso no terminal
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    checkpoint_path = f"CheckPoints/neat_{str(datetime.now())[:-7].replace(' ', '').replace(':','-')}checkpoint"
    p.add_reporter(
        neat.Checkpointer(generation_interval=1, filename_prefix=checkpoint_path)
    )
    if CarregarWin:
        winner_file = "genomavencedor/genoma_vencedor5.116135.pkl"
        with open(winner_file, "rb") as f:
            winner1 = pickle.load(f)
        winner_file = (
            "genomavencedor/genoma_vencedor19.8792.pkl"
        )
        with open(winner_file, "rb") as f:
            winner = pickle.load(f)
        # Ensure there is at least one species.
        if not p.species.species:
            # Create a new species if the list is empty
            p.species.create_new_species()
        # Replace the champion genome in the new population with the saved winner genome.
        if p.species.species:
            winner_dict = {winner.key: winner}
            winner_dict = {winner.key: winner1}
            p.species.species[1].members = winner_dict
            p.species.species[2].members = winner_dict

    # Verifique se há um genoma vencedor pré-treinado para iniciar a população

    # Executar por até 300 gerações
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = p.run(pe.evaluate, num_generations)

    # Salvar o genoma vencedor
    with open(
        f"genomavencedor/genoma_vencedor{str(datetime.now())[:-7].replace(' ', '').replace(':','-')}.pkl",
        "wb",
    ) as arquivo:
        pickle.dump(winner, arquivo)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "configteste.txt")
    run(config_path)
    # cProfile.run('run(config_path)')
