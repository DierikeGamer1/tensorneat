from pipeline import Pipeline
from algorithm.neat import *
from datetime import datetime
import pickle
import utils 
import json
import sys 
import dill
from problem.rl_env.rl_jit import RLEnv
from problem.func_fit import XOR3d
import jax.numpy as jnp
from utils import Act

if __name__ == "__main__":
    pipeline = Pipeline(
        algorithm=NEAT(
            species=DefaultSpecies(
                genome=DefaultGenome(
                    num_inputs=202,
                    num_outputs=3,
                    max_nodes=500,
                    max_conns=500,
                    node_gene=DefaultNodeGene(
                        activation_options=(Act.tanh,),
                        activation_default=Act.tanh,
                    ),
                    output_transform=lambda out: jnp.argmax(out)
                ),
                pop_size=1000,
                species_size=10,
                compatibility_threshold=3.5,
            ),
        ),
        problem=RLEnv(),
        generation_limit=5,
        fitness_target=30,
    )

    # initialize state
    state = pipeline.setup()
    # with open("estado_inicial.json", "wb") as arquivo:
    #     dill.dump(state, arquivo)
    # with open("estado_inicial.json", "rb") as arquivo:
    #     state = dill.load(arquivo)
    # run until terminate
    state, best = pipeline.auto_run(state)
    # with open("estado_inicial.json", "wb") as arquivo:
    #     dill.dump(state, arquivo)
    # show result
    pipeline.show(state, best)
