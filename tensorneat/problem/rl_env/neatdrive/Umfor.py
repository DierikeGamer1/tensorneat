import neat
import json
from datetime import datetime
import multiprocessing
from multiprocessing import Pool
import os
import time
import classe
import sys
import pickle
import cProfile

num_generations = 500
num_episodes = 1
cont = 0
ListaVchloE = []
ListaClosePrice = []
Listadados = []
ListaGenoma = []
ListaGenomasEmTraining = []
ListaGenomasEmTrainingAntiga = []

Printar = False


def VerificarFinals(Final, indice, ListaClosePrice, env, SellPrice):
    reward = 0
    if Final:
        reward = env.FecharAcabouDados(SellPrice)
    elif indice == ListaClosePrice-1:
        Final = True
        reward = env.FecharAcabouDados(SellPrice)
    return Final, reward


def argmax_custom(lst):
    max_index = 0
    max_value = float('-inf')
    for i, val in enumerate(lst):
        if val > max_value:
            max_value = val
            max_index = i
    return max_index


def RetornaAtivado(genome_id, listG):
    return argmax_custom(listG[genome_id].activate(listG[genome_id].state))


def DividirGenomas(ListaGenomas, ListaIp=10):
    # Calcula o tamanho de cada parte
    tamanho_parte = len(ListaGenomas) // ListaIp
    sobra = len(ListaGenomas) % ListaIp  # Calcula a sobra

    # Lista para armazenar as partes para cada pessoa
    partes_por_pessoa = []

    inicio = 0
    for i in range(ListaIp):
        # Calcula o fim do intervalo para esta pessoa
        fim = inicio + tamanho_parte + (1 if i < sobra else 0)

        # Adiciona a parte desta pessoa à lista de partes
        partes_por_pessoa.append(ListaGenomas[inicio:fim])

        # Atualiza o início para a próxima pessoa
        inicio = fim

    # Exibe as partes para cada pessoa
    Genomas = []
    for i, partes in enumerate(partes_por_pessoa):
        # print(f'Ip {ListaIp[i-1]}: {partes}')
        Genomas.append(partes)
    return Genomas


def eval_genome(genomes, config, results_dict, index):
    global Printar
    listG = []
    for indice, (genome_id, g) in enumerate(genomes):
        listG.append(neat.nn.FeedForwardNetwork.create(g, config))
        listG[indice].total_reward = 0
        listG[indice].fitness = 0
        listG[indice].episode_reward = 0
        listG[indice].contLong = 0
        listG[indice].contShort = 0

    lista = os.listdir("DadosTreino")
    for numero, nomearquivo in enumerate(lista):
        if "ada" in nomearquivo:
            if Printar:
                print(nomearquivo)
            # print(nomearquivo)
            with open(f"DadosTreino/{nomearquivo}", "r") as nomearquivo:
                Listadados = []
                Listadados.append([json.loads(linha.replace("'", '"').replace(
                    "+", ""))["data"]for linha in nomearquivo])

                ListaClosePrice = []
                ListaClosePrice.append(
                    [float(objeto["c"])
                        for objeto in Listadados[len(Listadados) - 1]]
                )

            for arquivo in ListaClosePrice:
                ListaGenomasEmTrainingAntiga = genomes
                ClosePrice = arquivo

                genomes_dict = {(genome_id): index for index,
                                (genome_id, g) in enumerate(genomes)}
                # Move a inicialização do ambiente e do estado para fora do loop interno
                for indice, (genome_id, g) in enumerate(genomes):
                    listG[indice].env = classe.AmbientDeTreino(ClosePrice)
                    listG[indice].state = listG[indice].env.reset()

                for indice1 in range(len(Listadados[0])):
                    # Fecha Caso Nao Tenha Mais Genomas Em Treinamento
                    if len(ListaGenomasEmTrainingAntiga) == 0:
                        if Printar:
                            print(
                                "Maior que 206 e sem treino!\n---------Erro--------")
                            print(indice1)
                        break
                    # Printar Quantos Vivos Em cada 1000 Indices
                    # if str(indice1)[-3:] == '000':
                    #     if Printar:
                    #         print(indice1)
                    #         print(f"vivos:{len(ListaGenomasEmTraining)}")

                    if indice1 > 202:
                        ListaGenomasEmTraining = ListaGenomasEmTrainingAntiga
                        ListaGenomasEmTrainingAntiga = []
                        
                        start = time.time()
                        for i in range(1000):
                            actions = []
                            for genome_id, g in ListaGenomasEmTraining:
                                index = genomes_dict.get(genome_id)
                                actions.append(argmax_custom(listG[index].activate(listG[index].state)))  # type: ignore
                        print(time.time()-start)
                        sys.exit()
                        # start=time.time()
                        # for i in range(400000):
                        for genome_id, g in ListaGenomasEmTraining:
                            index = genomes_dict.get(genome_id)
                            actions.append(
                                argmax_custom(listG[index].activate(listG[index].state)))  # type: ignore
                        # print(f"Tempo:{time.time()-start}")
                        # sys.exit()
                        closepriceExtend = ClosePrice[indice1 - 200: indice1]
                        UltimosDoisFechamentos = ClosePrice[indice1-1:indice1+1]

                        for indice, (genome_id, g) in enumerate(ListaGenomasEmTraining):
                            action = actions[indice]
                            genoma_index = genomes_dict.get(genome_id)
                            reward, Final, listG[genoma_index].state = listG[genoma_index].env.actions(  # type: ignore
                                action, UltimosDoisFechamentos)
                            if action == 0:
                                # type: ignore
                                listG[genoma_index].contLong += 1
                            elif action == 2:
                                # type: ignore
                                listG[genoma_index].contShort += 1
                            listG[genoma_index].state.extend(  # type: ignore
                                closepriceExtend)
                            # type: ignore
                            listG[genoma_index].episode_reward += reward

                            listG[genoma_index].final, reward1 = VerificarFinals(Final, indice1, len(ListaClosePrice),  # type: ignore
                                                                                 listG[genoma_index].env, UltimosDoisFechamentos)
                            if listG[genoma_index].final == True or Final == True:  # type: ignore
                                # print(listG[genoma_index].final)
                                # sys.exit()
                                # type: ignore
                                listG[genoma_index].episode_reward += reward1
                            else:
                                ListaGenomasEmTrainingAntiga.append(
                                    (genome_id, g))
                                if Printar:
                                    print("Foi Para Proxima Geracao")
                                    print(len(ListaGenomasEmTrainingAntiga))
    sys.exit()
    for genome_id, g in genomes:
        genoma_index = genomes.index((genome_id, g))
        g.fitness = listG[genoma_index].episode_reward
        if g.fitness > 0.005:
            if listG[genoma_index].contLong > 0 and listG[genoma_index].contShort > 0:
                print(f"ContiLong:{listG[genoma_index].contLong}")
                print(f"ContiShort:{listG[genoma_index].contShort}")
                g.fitness = g.fitness+2
                print(g.fitness)
        # results_dict[genome_id] = g.fitness
    print(f"acabou:{len(genomes)}")


def main(genomes, config):
    # processes = []
    # ListaGenomas = DividirGenomas(genomes[:8], multiprocessing.cpu_count()*2)
    # results_dict = multiprocessing.Manager().dict()

    # for i in range(len(ListaGenomas)):
    #     process = multiprocessing.Process(target=eval_genome, args=(
    #         ListaGenomas[i], config, results_dict, i))
    #     processes.append(process)
    #     process.start()
    # for process in processes:
    #     process.join()

    # for genomeid, g in genomes:
    #     g.fitness = results_dict.get(genomeid)
    # Teste Sig
    eval_genome(genomes, config, 'd', 'a')
    sys.exit()
    return genomes


def run(config_file, CarregarWin=False, CarregarCheckPoint=True):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )

    if CarregarCheckPoint:
        p = neat.Checkpointer.restore_checkpoint(
            "CheckPoints/neat_2024-03-0521-04-17checkpoint366"
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
        winner_file = "genomavencedor/bestGenome2.27219999999999932024-02-2513-17-57.pkl"
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

    winner = p.run(main, num_generations)

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
    # cProfile.run('run(config_path)')
