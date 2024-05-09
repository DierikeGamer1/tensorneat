import neat
import classe
import os
import json


def eval_genome(genome, config, num_episodes, ListaClosePrice, Listadados):
    total_reward = 0
    fitness = 0
    episode_reward = 0
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    for numero, arquivo in enumerate(ListaClosePrice):
        ClosePrice = ListaClosePrice[numero]
        dados = Listadados[numero]
        env = classe.AmbientDeTreino(ClosePrice)
        for _ in range(num_episodes):
            state = env.reset()
            for indice, objeto in enumerate(dados):
                if indice > 201:
                    newClosePrice = ClosePrice[indice]
                    action_output = net.activate(state)
                    action = np.argmax(action_output)

                    reward, Final, PositionLong, PositionShort = env.actions(
                        action, newClosePrice
                    )
                    if Final:
                        break
                    elif episode_reward < -50:
                        break

                    state = [
                        float(PositionLong),
                        float(PositionShort),
                        *ClosePrice[indice - 200 : indice],
                    ]

                    episode_reward += reward

            total_reward += episode_reward

    fitness += total_reward / (num_episodes)
    return fitness


def ImportarDados(Content=None):
    lista = os.listdir("DadosTreino")
    ListaClosePrice = []
    Listadados = []
    VchloE = []
    for numero, nomearquivo in enumerate(lista):

        if Content == nomearquivo:
            with open(f"DadosTreino/{nomearquivo}", "r") as arquivo:
                Listadados = [
                    json.loads(linha.replace("'", '"').replace("+", ""))["data"]
                    for linha in arquivo
                ]
                ListaClosePrice = [float(objeto["c"]) for objeto in Listadados]
                VchloE = [
                    [
                        float(objeto["v"]),
                        float(objeto["c"]),
                        float(objeto["h"]),
                        float(objeto["l"]),
                        float(objeto["o"]),
                        float(objeto["E"]),
                    ]
                    for objeto in Listadados
                ]

        

    return ListaClosePrice, Listadados, VchloE
