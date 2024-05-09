
import time 
start=time.time()
with open("Processados.txt",'r') as arquivo:
    c=arquivo.read()
c=[float(v.replace('[',"").replace(']',"").replace(',',"")) for v in c.split()]
print(len(c))
1069904
# soma= 0
# for v in c :
#     soma+=v
# print(soma)
# # import re

# # soma = 0
# # with open("Processados.txt", 'r') as arquivo:
# #     for linha in arquivo:
# #         valores = re.findall(r'[-+]?\d*\.\d+|\d+', linha)  # Encontra todos os números na linha
# #         soma += sum(map(float, valores))

# # print(soma)
# print(time.time()-start)