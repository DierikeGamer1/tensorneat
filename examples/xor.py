import jax
import numpy as np

from config import Config, BasicConfig, NeatConfig
from pipeline import Pipeline
from algorithm import NEAT
from algorithm.neat.gene import NormalGene, NormalGeneConfig

xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
xor_outputs = np.array([[0], [1], [1], [0]], dtype=np.float32)


def evaluate(forward_func):
    """
    :param forward_func: (4: batch, 2: input size) -> (pop_size, 4: batch, 1: output size)
    :return:
    """
    outs = forward_func(xor_inputs)
    outs = jax.device_get(outs)
    fitnesses = 4 - np.sum((outs - xor_outputs) ** 2, axis=(1, 2))
    return fitnesses


if __name__ == '__main__':
    config = Config(
        basic=BasicConfig(
            fitness_target=3.9999999,
            pop_size=10000
        ),
        neat=NeatConfig(
            maximum_nodes=20,
            maximum_conns=50,
        )
    )
    normal_gene = NormalGene(NormalGeneConfig())
    algorithm = NEAT(config, normal_gene)
    pipeline = Pipeline(config, algorithm)
    pipeline.auto_run(evaluate)
