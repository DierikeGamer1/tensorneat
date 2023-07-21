from typing import Type

import jax
import jax.numpy as jnp

from algorithm import Algorithm, State
from .gene import BaseGene
from .genome import initialize_genomes
from .population import create_tell


class NEAT(Algorithm):
    def __init__(self, config, gene_type: Type[BaseGene]):
        super().__init__()
        self.config = config
        self.gene_type = gene_type

        self.tell = create_tell(config, self.gene_type)
        self.ask = None
        self.forward = self.gene_type.create_forward(config)
        self.forward_transform = self.gene_type.forward_transform

    def setup(self, randkey, state=State()):
        state = state.update(
            P=self.config['pop_size'],
            N=self.config['maximum_nodes'],
            C=self.config['maximum_conns'],
            S=self.config['maximum_species'],
            NL=1 + len(self.gene_type.node_attrs),  # node length = (key) + attributes
            CL=3 + len(self.gene_type.conn_attrs),  # conn length = (in, out, key) + attributes
            input_idx=self.config['input_idx'],
            output_idx=self.config['output_idx'],
            max_stagnation=self.config['max_stagnation'],
            species_elitism=self.config['species_elitism'],
            spawn_number_change_rate=self.config['spawn_number_change_rate'],
            genome_elitism=self.config['genome_elitism'],
            survival_threshold=self.config['survival_threshold'],
            compatibility_threshold=self.config['compatibility_threshold'],
        )

        state = self.gene_type.setup(state, self.config)

        randkey = randkey
        pop_nodes, pop_conns = initialize_genomes(state, self.gene_type)
        species_info = jnp.full((state.S, 4), jnp.nan,
                                dtype=jnp.float32)  # (species_key, best_fitness, last_improved, size)
        species_info = species_info.at[0, :].set([0, -jnp.inf, 0, state.P])
        idx2species = jnp.zeros(state.P, dtype=jnp.float32)
        center_nodes = jnp.full((state.S, state.N, state.NL), jnp.nan, dtype=jnp.float32)
        center_conns = jnp.full((state.S, state.C, state.CL), jnp.nan, dtype=jnp.float32)
        center_nodes = center_nodes.at[0, :, :].set(pop_nodes[0, :, :])
        center_conns = center_conns.at[0, :, :].set(pop_conns[0, :, :])
        generation = 0
        next_node_key = max(*state.input_idx, *state.output_idx) + 2
        next_species_key = 1

        state = state.update(
            randkey=randkey,
            pop_nodes=pop_nodes,
            pop_conns=pop_conns,
            species_info=species_info,
            idx2species=idx2species,
            center_nodes=center_nodes,
            center_conns=center_conns,

            # avoid jax auto cast from int to float. that would cause re-compilation.
            generation=jnp.asarray(generation, dtype=jnp.int32),
            next_node_key=jnp.asarray(next_node_key, dtype=jnp.float32),
            next_species_key=jnp.asarray(next_species_key, dtype=jnp.float32),
        )

        # move to device
        state = jax.device_put(state)

        return state