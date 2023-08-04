import jax.numpy as jnp

from config import *
from pipeline import Pipeline
from algorithm import NEAT
from algorithm.neat.gene import NormalGene, NormalGeneConfig
from problem.rl_env import GymNaxConfig, GymNaxEnv


def example_conf1():
    return Config(
        basic=BasicConfig(
            seed=42,
            fitness_target=500,
            pop_size=10000
        ),
        neat=NeatConfig(
            inputs=4,
            outputs=1,
        ),
        gene=NormalGeneConfig(
            activation_default='sigmoid',
            activation_options=('sigmoid',),
        ),
        problem=GymNaxConfig(
            env_name='CartPole-v1',
            output_transform=lambda out: jnp.where(out[0] > 0.5, 1, 0)  # the action of cartpole is {0, 1}
        )
    )


def example_conf2():
    return Config(
        basic=BasicConfig(
            seed=42,
            fitness_target=500,
            pop_size=10000
        ),
        neat=NeatConfig(
            inputs=4,
            outputs=1,
        ),
        gene=NormalGeneConfig(
            activation_default='tanh',
            activation_options=('tanh',),
        ),
        problem=GymNaxConfig(
            env_name='CartPole-v1',
            output_transform=lambda out: jnp.where(out[0] > 0, 1, 0)  # the action of cartpole is {0, 1}
        )
    )


def example_conf3():
    return Config(
        basic=BasicConfig(
            seed=42,
            fitness_target=500,
            pop_size=10000
        ),
        neat=NeatConfig(
            inputs=4,
            outputs=2,
        ),
        gene=NormalGeneConfig(
            activation_default='tanh',
            activation_options=('tanh',),
        ),
        problem=GymNaxConfig(
            env_name='CartPole-v1',
            output_transform=lambda out: jnp.argmax(out)  # the action of cartpole is {0, 1}
        )
    )


if __name__ == '__main__':
    # all config files above can solve cartpole
    conf = example_conf3()

    algorithm = NEAT(conf, NormalGene)
    pipeline = Pipeline(conf, algorithm, GymNaxEnv)
    state = pipeline.setup()
    state, best = pipeline.auto_run(state)
    pipeline.show(state, best)
