import neat
import pickle
import multiprocessing
from datetime import datetime
import classeshort
import os 
import cython


config_path = (
    "configacao.txt"  # Substitua pelo caminho real do seu arquivo de configuração
)
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path,
)
CarregarVencedor = True
contLong = 0
contShort = 0
num_episodes = 1

def VerificarFinals(Final, indice, ListaClosePrice, env, SellPrice):
    reward = 0
    if Final:
        reward = env.FecharAcabouDados(SellPrice,indice)
    elif indice == ListaClosePrice-1:
        Final = True
        reward = env.FecharAcabouDados(SellPrice,indice)
    return Final, reward

def argmax_custom(lst):
    max_index, _ = max(enumerate(lst), key=lambda x: x[1])
    return max_index



# Carregando a rede neural vencedora
if CarregarVencedor:
    with open("genomavencedor/genoma_vencedor19.70120000000001.pkl", "rb") as f:
        winning_net = pickle.load(f)
    net = neat.nn.FeedForwardNetwork.create(winning_net, config)
else:
    pass
    # p = neat.Checkpointer.restore_checkpoint(
    #     "CheckPoints/neat_2024-01-3122-11-39checkpoint40"
    # )
    # pe = neat.ParallelEvaluator(multiprocessing.cpu_count(),eval_genome)
    # winner = p.run(pe.evaluate, 1)

    # # Criar a rede neural a partir do melhor genoma
    # winning_net = neat.nn.FeedForwardNetwork.create(winner, config)

contLong:cython.int = 0
contShort:cython.int = 0
total_reward:cython.int = 0
fitness:cython.float = 0
episode_reward:cython.float = 0
lista = os.listdir("DadosTreino")
for numero, nomearquivo in enumerate(lista):
    if "adausdt2024-02-20-00-21-41-425423" in nomearquivo:
        with open("Processados.txt",'r') as arquivo:
            ListaClosePrice=[[float(v.replace('[',"").replace(']',"").replace(',',"")) for v in arquivo.read().split()]]
            
            for arquivo in ListaClosePrice:
                ClosePrice = arquivo
                dados = ListaClosePrice
                env = classeshort.AmbientDeTreino(ClosePrice)
                state = env.reset()
                for indice in range(len(dados[0])):                
                    if indice > 202:                          
                        action = argmax_custom(net.activate(state))
                        reward, Final, state = env.actions(action, ClosePrice)
                        if action == 0:
                            # contLong += 1
                            pass
                        elif action == 2:
                            contShort += 1
                        # if action != 1 :
                        #     print(action)
                        #     print(f"indice:{indice}")
                        #     print(f"ContiLong:{contLong}")
                        #     print(f"ContiShort:{contShort}")
                        #     print(f"reward:{reward}")
                            # print(datetime.fromtimestamp((VchloE[indice][5]) / 1000))

                        state.extend(ClosePrice[indice - 200: indice])
                        episode_reward += reward
                        
                        final, reward1 = VerificarFinals(
                            Final, indice, len(ListaClosePrice), env, ClosePrice)      
                        if final or Final:
                            episode_reward += reward1
                            print(indice)
                            break
            fitness += episode_reward
        
        if Final:
            print("break")
            break

print(f"ContiLong:{contLong}")
print(f"ContiShoort:{contShort}")
print(f"reward total:{fitness}")
