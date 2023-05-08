from typing import List, Union, Tuple, Callable
import time

import jax
import jax.numpy as jnp
import numpy as np

from .species import SpeciesController
from .genome import expand, expand_single
from .genome import create_initialize_function, create_mutate_function, create_forward_function, \
    create_distance_function, create_crossover_function


class Pipeline:
    """
    Neat algorithm pipeline.
    """

    def __init__(self, config, seed=42):
        self.generation_timestamp = time.time()
        self.randkey = jax.random.PRNGKey(seed)
        np.random.seed(seed)

        self.config = config
        self.N = config.basic.init_maximum_nodes
        self.expand_coe = config.basic.expands_coe
        self.pop_size = config.neat.population.pop_size

        self.species_controller = SpeciesController(config)
        self.initialize_func = create_initialize_function(config)
        self.pop_nodes, self.pop_connections, self.input_idx, self.output_idx = self.initialize_func()

        self.compile_functions(debug=True)

        self.generation = 0
        self.species_controller.init_speciate(self.pop_nodes, self.pop_connections)

        self.best_fitness = float('-inf')

    def ask(self, batch: bool):
        """
        Create a forward function for the population.
        :param batch:
        :return:
        Algorithm gives the population a forward function, then environment gives back the fitnesses.
        """
        func = create_forward_function(self.pop_nodes, self.pop_connections, self.N, self.input_idx, self.output_idx,
                                       batch=batch)
        return func

    def tell(self, fitnesses):

        self.generation += 1

        self.species_controller.update_species_fitnesses(fitnesses)

        crossover_pair = self.species_controller.reproduce(self.generation)

        self.update_next_generation(crossover_pair)

        self.species_controller.speciate(self.pop_nodes, self.pop_connections, self.generation,
                                         self.o2o_distance, self.o2m_distance)

        self.expand()

    def auto_run(self, fitness_func, analysis: Union[Callable, str] = "default"):
        for _ in range(self.config.neat.population.generation_limit):
            forward_func = self.ask(batch=True)
            fitnesses = fitness_func(forward_func)

            if analysis is not None:
                if analysis == "default":
                    self.default_analysis(fitnesses)
                else:
                    assert callable(analysis), f"What the fuck you passed in? A {analysis}?"
                    analysis(fitnesses)

            self.tell(fitnesses)
        print("Generation limit reached!")

    def update_next_generation(self, crossover_pair: List[Union[int, Tuple[int, int]]]) -> None:
        """
        create the next generation
        :param crossover_pair: created from self.reproduce()
        """

        assert self.pop_nodes.shape[0] == self.pop_size
        k1, k2, self.randkey = jax.random.split(self.randkey, 3)

        # crossover
        # prepare elitism mask and crossover pair
        elitism_mask = np.full(self.pop_size, False)

        for i, pair in enumerate(crossover_pair):
            if not isinstance(pair, tuple):  # elitism
                elitism_mask[i] = True
                crossover_pair[i] = (pair, pair)
        crossover_pair = np.array(crossover_pair)

        crossover_rand_keys = jax.random.split(k1, self.pop_size)
        # batch crossover
        wpn = self.pop_nodes[crossover_pair[:, 0]]  # winner pop nodes
        wpc = self.pop_connections[crossover_pair[:, 0]]  # winner pop connections
        lpn = self.pop_nodes[crossover_pair[:, 1]]  # loser pop nodes
        lpc = self.pop_connections[crossover_pair[:, 1]]  # loser pop connections
        npn, npc = self.crossover_func(crossover_rand_keys, wpn, wpc, lpn,
                                       lpc)  # new pop nodes, new pop connections

        # mutate
        mutate_rand_keys = jax.random.split(k2, self.pop_size)
        new_node_keys = np.arange(self.generation * self.pop_size, self.generation * self.pop_size + self.pop_size)

        m_npn, m_npc = self.mutate_func(mutate_rand_keys, npn, npc, new_node_keys)  # mutate_new_pop_nodes

        # elitism don't mutate
        npn, npc, m_npn, m_npc = jax.device_get([npn, npc, m_npn, m_npc])
        self.pop_nodes = np.where(elitism_mask[:, None, None], npn, m_npn)
        self.pop_connections = np.where(elitism_mask[:, None, None, None], npc, m_npc)

    def expand(self):
        """
        Expand the population if needed.
        :return:
        when the maximum node number of the population >= N
        the population will expand
        """
        pop_node_keys = self.pop_nodes[:, :, 0]
        pop_node_sizes = np.sum(~np.isnan(pop_node_keys), axis=1)
        max_node_size = np.max(pop_node_sizes)
        if max_node_size >= self.N:
            self.N = int(self.N * self.expand_coe)
            print(f"expand to {self.N}!")
            self.pop_nodes, self.pop_connections = expand(self.pop_nodes, self.pop_connections, self.N)

            # don't forget to expand representation genome in species
            for s in self.species_controller.species.values():
                s.representative = expand_single(*s.representative, self.N)

            # update functions
            self.compile_functions(debug=True)

    def compile_functions(self, debug=False):
        self.mutate_func = create_mutate_function(self.N, self.config, batch=True, debug=debug)
        self.crossover_func = create_crossover_function(self.N, self.config, batch=True, debug=debug)
        self.o2o_distance = create_distance_function(self.N, self.config, type='o2o', debug=debug)
        self.o2m_distance = create_distance_function(self.N, self.config, type='o2m', debug=debug)

    def default_analysis(self, fitnesses):
        max_f, min_f, mean_f, std_f = max(fitnesses), min(fitnesses), np.mean(fitnesses), np.std(fitnesses)
        species_sizes = [len(s.members) for s in self.species_controller.species.values()]

        new_timestamp = time.time()
        cost_time = new_timestamp - self.generation_timestamp
        self.generation_timestamp = new_timestamp

        print(f"Generation: {self.generation}",
              f"fitness: {max_f}, {min_f}, {mean_f}, {std_f}, Species sizes: {species_sizes}, Cost time: {cost_time}")
