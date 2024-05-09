import multiprocessing
import time

def worker(data):
    # Função que realiza a carga de trabalho
    return sum(data)

if __name__ == "__main__":
    # Dados de entrada (lista grande de números)
    data = list(range(50000000))

    # Número de processos a serem criados (usando o número de CPUs disponíveis)
    num_processes = 2

    print(f"Running with {num_processes} processes...")

    start_time = time.time()

    # Criar processos
    processes = []
    chunk_size = len(data) // num_processes
    for i in range(num_processes):
        start = i * chunk_size
        end = start + chunk_size if i < num_processes - 1 else len(data)
        process = multiprocessing.Process(target=worker, args=(data[start:end],))
        processes.append(process)
        process.start()

    # Aguardar que todos os processos terminem
    for process in processes:
        process.join()

    end_time = time.time()

    print(f"Total time taken: {end_time - start_time} seconds")
