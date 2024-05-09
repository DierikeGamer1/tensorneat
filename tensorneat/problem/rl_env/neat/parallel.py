"""
Runs evaluation functions in parallel subprocesses
in order to evaluate multiple genomes at once.
"""

from multiprocessing import Pool
import os, json, sys


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
        self.pool.close()  # should this be terminate?
        self.pool.join()

    def evaluate(self, genomes, config):
        nome_arquivo = sys.argv[0]
        jobs = []

        if (
            "ContinuarTreinandoTemporario.py" in nome_arquivo
            or "ContinuarShort.py" in nome_arquivo
            or "testejax.py" in nome_arquivo
        ):

            lista = os.listdir("./neatdrive/DadosTreino")
            ListaClosePrice = []
            for numero, nomearquivo in enumerate(lista):
                if "adausdt" in nomearquivo:
                    if "adausdt2024-02-20-00-21-41-425423" in nomearquivo:
                        with open(
                            f"neatdrive/DadosTreino/{nomearquivo}", "r"
                        ) as nomearquivo:
                            Listadados = []
                            Listadados.append(
                                [
                                    json.loads(
                                        linha.replace("'", '"').replace("+", "")
                                    )["data"]
                                    for linha in nomearquivo
                                ]
                            )
                            ListaClosePrice.append(
                                [
                                    float(objeto["c"])
                                    for objeto in Listadados[len(Listadados) - 1]
                                ]
                            )
                            t_DeInterval = 30000
                            div = int(
                                len(ListaClosePrice[len(ListaClosePrice) - 1])
                                / t_DeInterval
                            )
                            cgh = len(ListaClosePrice) - 1
                            for i in range(1, int(div)):
                                ListaClosePrice.append(
                                    ListaClosePrice[cgh][
                                        t_DeInterval * i
                                        + 1
                                        - t_DeInterval : t_DeInterval * i
                                    ]
                                )

                    else:
                        with open(
                            f"neatdrive/DadosTreino/{nomearquivo}", "r"
                        ) as nomearquivo:
                            Listadados = []
                            Listadados.append(
                                [
                                    json.loads(
                                        linha.replace("'", '"').replace("+", "")
                                    )["data"]
                                    for linha in nomearquivo
                                ]
                            )
                            ListaClosePrice.append(
                                [
                                    float(objeto["c"])
                                    for objeto in Listadados[len(Listadados) - 1]
                                ]
                            )

            for ignored_genome_id, genome in genomes:
                jobs.append(
                    self.pool.apply_async(
                        self.eval_function, (genome, config, ListaClosePrice)
                    )
                )

            # assign the fitness back to each genome
            for job, (ignored_genome_id, genome) in zip(jobs, genomes):
                genome.fitness = job.get(timeout=self.timeout)
        else:
            for ignored_genome_id, genome in genomes:
                jobs.append(self.pool.apply_async(self.eval_function, (genome, config)))
            # assign the fitness back to each genome
            for job, (ignored_genome_id, genome) in zip(jobs, genomes):
                genome.fitness = job.get(timeout=self.timeout)
